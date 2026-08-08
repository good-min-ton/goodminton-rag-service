import pytest

from app.routers.chat import SANITIZE_FALLBACK, _stream_turn


async def _fake(tokens, tool_calls=None):
    for t in tokens:
        yield ("token", t)
    yield ("final", {"role": "assistant", "content": "", "tool_calls": tool_calls})


async def _drain(tokens, tool_calls=None):
    live, result = [], None
    async for kind, val in _stream_turn(_fake(tokens, tool_calls)):
        if kind == "token":
            live.append(val)
        else:
            result = val
    return "".join(live), result


@pytest.mark.asyncio
async def test_prose_streams_live():
    live, result = await _drain(["Xin", " chào", " bạn"])
    assert live == "Xin chào bạn"
    assert result == ("answer", "Xin chào bạn")


@pytest.mark.asyncio
async def test_leading_whitespace_then_prose_streams():
    live, result = await _drain(["  ", "Chào", " bạn"])
    assert live == "  Chào bạn"
    assert result == ("answer", "  Chào bạn")


@pytest.mark.asyncio
async def test_json_blob_buffered_to_fallback_no_live():
    live, result = await _drain(['{"a"', ": 1}"])
    assert live == ""  # never streamed
    assert result == ("answer", SANITIZE_FALLBACK)


@pytest.mark.asyncio
async def test_recovered_tool_call_from_content_emits_no_tokens():
    live, result = await _drain(
        ['{"name": "get_product_availability", "arguments": {"product_id": 12}}']
    )
    assert live == ""
    assert result[0] == "tool"
    assert result[1] == [
        {
            "function": {
                "name": "get_product_availability",
                "arguments": {"product_id": 12},
            }
        }
    ]


@pytest.mark.asyncio
async def test_structured_tool_call_no_content():
    tc = [
        {
            "function": {
                "name": "get_product_availability",
                "arguments": {"product_id": 12},
            }
        }
    ]
    live, result = await _drain([], tool_calls=tc)
    assert live == ""
    assert result == ("tool", tc)


@pytest.mark.asyncio
async def test_fence_buffered_to_fallback():
    live, result = await _drain(['```json\n{"x": 1}\n```'])
    assert live == ""
    assert result == ("answer", SANITIZE_FALLBACK)


@pytest.mark.asyncio
async def test_lone_backtick_holds_then_returns_unchanged():
    # A single (non-triple) backtick start must still HOLD, not stream live.
    # Guards against narrowing the check to startswith("```").
    live, result = await _drain(["`", "code", "`"])
    assert live == ""  # never streamed
    assert result == ("answer", "`code`")  # _sanitize_answer leaves it unchanged


@pytest.mark.asyncio
async def test_whitespace_only_and_empty_fall_back():
    live, result = await _drain(["   "])
    assert live == ""
    assert result == ("answer", SANITIZE_FALLBACK)

    live, result = await _drain([])
    assert live == ""
    assert result == ("answer", SANITIZE_FALLBACK)


@pytest.mark.asyncio
async def test_live_prose_ignores_tool_calls_at_final():
    # Once prose has streamed live, the turn is an answer — tool_calls at final
    # are ignored (locks branch-order: decided=='live' wins before tool_calls).
    tc = [
        {
            "function": {
                "name": "get_product_availability",
                "arguments": {"product_id": 1},
            }
        }
    ]
    live, result = await _drain(["Xin chào"], tool_calls=tc)
    assert live == "Xin chào"
    assert result == ("answer", "Xin chào")
