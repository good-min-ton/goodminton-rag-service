import json
import httpx
import pytest
from app.services.llm import LLMService


def _client(lines: list[dict]):
    body = ("\n".join(json.dumps(ln) for ln in lines)).encode()

    async def handler(req):
        assert req.url.path == "/api/chat"
        return httpx.Response(200, content=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_stream_yields_tokens_then_final():
    lines = [
        {"message": {"role": "assistant", "content": "Xin"}, "done": False},
        {"message": {"role": "assistant", "content": " chào"}, "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True},
    ]
    async with _client(lines) as c:
        got = [e async for e in LLMService(c).chat_with_tools_stream([], [])]
    assert got == [
        ("token", "Xin"),
        ("token", " chào"),
        ("final", {"role": "assistant", "content": "", "tool_calls": None}),
    ]


@pytest.mark.asyncio
async def test_stream_surfaces_tool_calls_no_tokens():
    tc = [
        {
            "function": {
                "name": "get_product_availability",
                "arguments": {"product_id": 21},
            }
        }
    ]
    lines = [
        {
            "message": {"role": "assistant", "content": "", "tool_calls": tc},
            "done": False,
        },
        {"message": {"role": "assistant", "content": ""}, "done": True},
    ]
    async with _client(lines) as c:
        got = [e async for e in LLMService(c).chat_with_tools_stream([], [])]
    assert got == [("final", {"role": "assistant", "content": "", "tool_calls": tc})]


@pytest.mark.asyncio
async def test_stream_raises_on_non_200():
    # degrade contract: a failing Ollama stream must raise (endpoint maps to error event)
    async def handler(req):
        return httpx.Response(500)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
        with pytest.raises(httpx.HTTPStatusError):
            [e async for e in LLMService(c).chat_with_tools_stream([], [])]
