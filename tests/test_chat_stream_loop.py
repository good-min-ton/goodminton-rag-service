import json
import pytest
from app.routers.chat import _run_tool_loop_stream


async def _fake_turn(tokens, tool_calls=None):
    for t in tokens:
        yield ("token", t)
    yield ("final", {"role": "assistant", "content": "", "tool_calls": tool_calls})


class _FakeLLM:
    """chat_with_tools_stream replays queued turns; chat() = forced-final."""

    def __init__(self, turns):
        self._turns = list(turns)  # each: (tokens, tool_calls)
        self.forced = 0

    def chat_with_tools_stream(self, messages, tools):
        return _fake_turn(*self._turns.pop(0))

    async def chat(self, messages, **kw):
        self.forced += 1
        return "Xin lỗi, mình chưa tạo được đơn, bạn thử lại nhé."


class _Dispatcher:
    def __init__(self, results):
        self._results = results
        self.calls = []

    async def execute(self, name, arguments):
        self.calls.append(name)
        return self._results[name]


async def _run(llm, dispatcher):
    live, final, heartbeats = [], None, 0
    async for kind, val in _run_tool_loop_stream(llm, dispatcher, []):
        if kind == "token":
            live.append(val)
        elif kind == "heartbeat":
            heartbeats += 1
        elif kind == "final":
            final = val
    return "".join(live), final, heartbeats


@pytest.mark.asyncio
async def test_pricing_then_prepare_order_then_streamed_answer():
    tc_price = [{"function": {"name": "get_pricing", "arguments": {"product_id": 12}}}]
    tc_order = [
        {
            "function": {
                "name": "prepare_order",
                "arguments": {
                    "items": [{"product_id": 12, "variant_id": 45, "quantity": 2}]
                },
            }
        }
    ]
    draft = json.dumps(
        {
            "items": [{"product_id": "12", "variant_id": "45", "quantity": 2}],
            "total": 6400000.0,
            "currency": "VND",
            "warnings": [],
        }
    )
    llm = _FakeLLM(
        [([], tc_price), ([], tc_order), (["Mời bạn ", "bấm XÁC NHẬN."], None)]
    )
    disp = _Dispatcher(
        {
            "get_pricing": json.dumps({"productId": 12, "variants": []}),
            "prepare_order": draft,
        }
    )
    live, (answer, _tp, order_draft), heartbeats = await _run(llm, disp)
    assert live == "Mời bạn bấm XÁC NHẬN."
    assert answer == "Mời bạn bấm XÁC NHẬN."
    assert order_draft is not None and order_draft["total"] == 6400000.0
    assert disp.calls == ["get_pricing", "prepare_order"]
    assert heartbeats == 3  # one per loop turn: pricing, prepare_order, answer


@pytest.mark.asyncio
async def test_status_emitted_per_tool_call_including_cache_hits():
    """The UI's thinking indicator names the tool that is running, so every tool
    call must surface a status — a cache hit is still a step the user is waiting
    on. Here the repeated get_pricing yields a third status but only two
    dispatcher executions."""
    tc_price = [{"function": {"name": "get_pricing", "arguments": {"product_id": 12}}}]
    tc_inv = [
        {"function": {"name": "check_inventory", "arguments": {"variant_id": 45}}}
    ]
    llm = _FakeLLM(
        [
            ([], tc_price),
            ([], tc_inv),
            ([], tc_price),  # repeat -> cache hit
            (["Còn hàng nhé."], None),
        ]
    )
    disp = _Dispatcher(
        {
            "get_pricing": json.dumps({"productId": 12, "variants": []}),
            "check_inventory": json.dumps([{"store": "HQ", "quantity": 3}]),
        }
    )
    statuses, final = [], None
    async for kind, val in _run_tool_loop_stream(llm, disp, []):
        if kind == "status":
            statuses.append(val)
        elif kind == "final":
            final = val
    assert [s["tool"] for s in statuses] == [
        "get_pricing",
        "check_inventory",
        "get_pricing",
    ]
    assert disp.calls == ["get_pricing", "check_inventory"]  # 3rd was cached
    assert final[0] == "Còn hàng nhé."


@pytest.mark.asyncio
async def test_repeated_calls_force_final_buffered():
    tc = [{"function": {"name": "get_pricing", "arguments": {"product_id": 12}}}]
    llm = _FakeLLM([([], tc), ([], tc), ([], tc), ([], tc)])
    disp = _Dispatcher({"get_pricing": json.dumps({"productId": 12, "variants": []})})
    live, (answer, _tp, order_draft), heartbeats = await _run(llm, disp)
    assert order_draft is None
    assert llm.forced == 1  # forced-final path
    assert answer == "Xin lỗi, mình chưa tạo được đơn, bạn thử lại nhé."
    assert disp.calls == ["get_pricing"]  # repeats are cache hits
    # 4 loop turns run before the 3rd repeat trips forced-final (1 original call
    # + 3 repeats == MAX_REPEATED_CALLS), one heartbeat per turn.
    assert heartbeats == 4
