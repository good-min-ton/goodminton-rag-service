import json
from unittest.mock import AsyncMock

from app.routers.chat import _parse_order_draft, _run_tool_loop


def test_parse_order_draft_valid_dict():
    payload = json.dumps({"items": [], "total": 0.0, "currency": "VND", "warnings": []})
    assert _parse_order_draft(payload) == {
        "items": [],
        "total": 0.0,
        "currency": "VND",
        "warnings": [],
    }


def test_parse_order_draft_error_payload_returns_none():
    assert _parse_order_draft(json.dumps({"error": "boom"})) is None


def test_parse_order_draft_missing_items_returns_none():
    assert _parse_order_draft(json.dumps({"total": 0.0})) is None


def test_parse_order_draft_non_json_returns_none():
    assert _parse_order_draft("not json") is None


class _Dispatcher:
    """Minimal stub: returns a canned tool result per tool name."""

    def __init__(self, results: dict[str, str]) -> None:
        self._results = results
        self.calls: list[str] = []

    async def execute(self, name: str, arguments: dict) -> str:
        self.calls.append(name)
        return self._results[name]


async def test_run_tool_loop_pricing_then_prepare_order_yields_draft():
    # LLM: turn 1 -> get_pricing, turn 2 -> prepare_order, turn 3 -> text answer.
    llm = AsyncMock()
    llm.chat_with_tools.side_effect = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "get_pricing", "arguments": {"product_id": 12}}}
            ],
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "prepare_order",
                        "arguments": {
                            "items": [
                                {"product_id": 12, "variant_id": 45, "quantity": 2}
                            ]
                        },
                    }
                }
            ],
        },
        {
            "role": "assistant",
            "content": "Bạn chọn size 4U màu Đỏ, mời bấm XÁC NHẬN trên thẻ đơn hàng.",
            "tool_calls": [],
        },
    ]

    draft_json = json.dumps(
        {
            "items": [
                {
                    "product_id": "12",
                    "variant_id": "45",
                    "product_name": "Vợt Yonex Astrox 100ZZ",
                    "size": "4U",
                    "color": "Đỏ",
                    "quantity": 2,
                    "unit_price": 3200000.0,
                    "line_total": 6400000.0,
                    "in_stock": True,
                }
            ],
            "total": 6400000.0,
            "currency": "VND",
            "warnings": [],
        }
    )
    dispatcher = _Dispatcher(
        {
            "get_pricing": json.dumps(
                {"productId": 12, "productName": "X", "variants": []}
            ),
            "prepare_order": draft_json,
        }
    )

    answer, tool_products, order_draft = await _run_tool_loop(llm, dispatcher, [])

    assert order_draft is not None
    assert order_draft["items"][0]["variant_id"] == "45"
    assert order_draft["total"] == 6400000.0
    assert "XÁC NHẬN" in answer
    assert dispatcher.calls == ["get_pricing", "prepare_order"]


async def test_run_tool_loop_repeated_calls_force_final_without_draft():
    # Model gets stuck repeating the SAME get_pricing call. After
    # MAX_REPEATED_CALLS (=3) cache-hits the loop forces a tool-free final
    # answer. No prepare_order ran, so order_draft stays None AND the
    # forced-final answer must not claim a confirmation card exists.
    stuck = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"function": {"name": "get_pricing", "arguments": {"product_id": 12}}}
        ],
    }
    llm = AsyncMock()
    # 4 identical tool turns: 1 fresh + 3 cache-hits -> repeats reaches 3.
    llm.chat_with_tools.side_effect = [stuck, stuck, stuck, stuck]
    llm.chat.return_value = "Xin lỗi, mình chưa tạo được đơn, bạn thử lại giúp nhé."

    dispatcher = _Dispatcher(
        {
            "get_pricing": json.dumps(
                {"productId": 12, "productName": "X", "variants": []}
            )
        }
    )

    answer, tool_products, order_draft = await _run_tool_loop(llm, dispatcher, [])

    assert order_draft is None  # no prepare_order -> no draft
    assert llm.chat.await_count == 1  # forced-final path was taken
    assert answer == "Xin lỗi, mình chưa tạo được đơn, bạn thử lại giúp nhé."
    assert dispatcher.calls == [
        "get_pricing"
    ]  # repeats are cache hits, not re-executed
    assert (
        llm.chat_with_tools.await_count == 4
    )  # 1 fresh + 3 cache-hit repeats -> forced final at MAX_REPEATED_CALLS=3
