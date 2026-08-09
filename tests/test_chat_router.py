import json
from unittest.mock import AsyncMock

from app.routers.chat import (
    SANITIZE_FALLBACK,
    _run_tool_loop,
)
from app.services.order_flow import (
    BROWSING,
    ORDER_CONFIRMED,
    WAITING_CONFIRMATION,
    suppresses_recommendations,
)


def test_display_blanked_when_order_pending():
    display = [1, 2, 3]
    gated = [] if suppresses_recommendations(WAITING_CONFIRMATION) else display
    assert gated == []
    gated2 = [] if suppresses_recommendations(ORDER_CONFIRMED) else display
    assert gated2 == []
    gated3 = [] if suppresses_recommendations(BROWSING) else display
    assert gated3 == [1, 2, 3]


# Card-selection is now handled by _card_candidates + RerankService; see
# tests/test_card_candidates.py and tests/test_rerank.py.


class _Dispatcher:
    """Minimal stub: returns a canned tool result per tool name."""

    def __init__(self, results: dict[str, str]) -> None:
        self._results = results
        self.calls: list[str] = []

    async def execute(self, name: str, arguments: dict) -> str:
        self.calls.append(name)
        return self._results[name]


async def test_run_tool_loop_start_order_yields_a_selection():
    """Buying is one tool call now: the model names the product and the picker
    takes over. It never sees a variant_id, so it cannot pick the wrong one."""
    llm = AsyncMock()
    llm.chat_with_tools.side_effect = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "start_order", "arguments": {"product_id": 12}}}
            ],
        },
        {
            "role": "assistant",
            "content": "Mời bạn chọn size và màu ngay bên dưới nhé.",
            "tool_calls": [],
        },
    ]

    selection_json = json.dumps(
        {
            "product_id": "12",
            "product_name": "Vợt Yonex Astrox 100ZZ",
            "currency": "VND",
            "options": [
                {
                    "variant_id": "45",
                    "size": "4U",
                    "color": "Đỏ",
                    "unit_price": 3200000.0,
                    "orderable": 5,
                    "branches": [],
                }
            ],
        }
    )
    dispatcher = _Dispatcher({"start_order": selection_json})

    answer, tool_products, order_selection = await _run_tool_loop(llm, dispatcher, [])

    assert order_selection is not None
    assert order_selection["options"][0]["variant_id"] == "45"
    assert order_selection["options"][0]["orderable"] == 5
    assert "chọn" in answer
    assert dispatcher.calls == ["start_order"]


async def test_run_tool_loop_repeated_calls_force_final_without_a_card():
    # Model gets stuck repeating the SAME get_product_availability call. After
    # MAX_REPEATED_CALLS (=3) cache-hits the loop forces a tool-free final
    # answer. No start_order ran, so no picker reaches the customer AND the
    # forced-final answer must not claim a card exists.
    stuck = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "get_product_availability",
                    "arguments": {"product_id": 12},
                }
            }
        ],
    }
    llm = AsyncMock()
    # 4 identical tool turns: 1 fresh + 3 cache-hits -> repeats reaches 3.
    llm.chat_with_tools.side_effect = [stuck, stuck, stuck, stuck]
    llm.chat.return_value = "Xin lỗi, mình chưa tạo được đơn, bạn thử lại giúp nhé."

    dispatcher = _Dispatcher(
        {
            "get_product_availability": json.dumps(
                {"productId": 12, "productName": "X", "variants": []}
            )
        }
    )

    answer, tool_products, order_selection = await _run_tool_loop(llm, dispatcher, [])

    assert order_selection is None  # no start_order -> no picker
    assert llm.chat.await_count == 1  # forced-final path was taken
    assert answer == "Xin lỗi, mình chưa tạo được đơn, bạn thử lại giúp nhé."
    assert dispatcher.calls == [
        "get_product_availability"
    ]  # repeats are cache hits, not re-executed
    assert (
        llm.chat_with_tools.await_count == 4
    )  # 1 fresh + 3 cache-hit repeats -> forced final at MAX_REPEATED_CALLS=3


async def test_tool_loop_recovers_tool_call_from_content():
    # Turn 1: model emits a get_product_availability call as JSON *content* (tool_calls empty).
    # Turn 2: model gives a natural-language answer. The loop must recover+execute
    # the call and return the clean answer — never the raw JSON.
    llm = AsyncMock()
    llm.chat_with_tools.side_effect = [
        {
            "role": "assistant",
            "content": '{"name": "get_product_availability", "arguments": {"product_id": 12}}',
            "tool_calls": [],
        },
        {
            "role": "assistant",
            "content": "Vợt Astrox 12 giá 1.200.000đ ạ.",
            "tool_calls": [],
        },
    ]
    dispatcher = _Dispatcher(
        {"get_product_availability": json.dumps({"productId": 12, "variants": []})}
    )
    answer, _, _ = await _run_tool_loop(llm, dispatcher, [])
    assert answer == "Vợt Astrox 12 giá 1.200.000đ ạ."
    assert dispatcher.calls == [
        "get_product_availability"
    ]  # recovered call was executed
    assert "{" not in answer


async def test_tool_loop_sanitizes_unrecoverable_json_content():
    # Model emits bare args (no tool name) as content and no tool_calls -> not
    # recoverable -> the answer is sanitized to the fallback, not the raw JSON.
    llm = AsyncMock()
    llm.chat_with_tools.return_value = {
        "role": "assistant",
        "content": '{"product_id": 164, "size": "M", "quantity": 1}',
        "tool_calls": [],
    }
    dispatcher = _Dispatcher({})
    answer, _, _ = await _run_tool_loop(llm, dispatcher, [])
    assert answer == SANITIZE_FALLBACK
