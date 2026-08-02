"""Ollama chat wrapper. Phase 3: non-streaming, no tool calling."""

import json

import httpx

from app.core.config import settings


class LLMService:
    def __init__(self, client: httpx.AsyncClient):
        self._client = client

    async def chat(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        num_predict: int | None = None,
        model: str | None = None,
    ) -> str:
        """Plain chat (no tools).

        Keyword-only overrides let callers (e.g. the description feature)
        use a different model/temperature/num_predict than the default
        chat settings. When omitted, behavior is unchanged.
        """
        options: dict = {
            "temperature": settings.llm_temperature
            if temperature is None
            else temperature
        }
        if num_predict is not None:
            options["num_predict"] = num_predict
        body = {
            "model": settings.llm_model if model is None else model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        r = await self._client.post(
            f"{settings.ollama_url}/api/chat",
            json=body,
            timeout=settings.llm_timeout_seconds,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]

    async def chat_with_tools(self, messages: list[dict], tools: list[dict]) -> dict:
        """Chat with tool calling. Returns full message dict (may contain tool_calls)."""
        r = await self._client.post(
            f"{settings.ollama_url}/api/chat",
            json={
                "model": settings.llm_model,
                "messages": messages,
                "tools": tools,
                "stream": False,
                "options": {"temperature": settings.llm_temperature},
            },
            timeout=settings.llm_timeout_seconds,
        )
        r.raise_for_status()
        return r.json()["message"]

    async def chat_with_tools_stream(self, messages: list[dict], tools: list[dict]):
        """Stream a tool-enabled chat turn. Async-generator: yields ("token", delta)
        per content delta, then exactly one ("final", message-dict) carrying the
        turn's tool_calls (list or None). Mirrors chat_with_tools() but stream=True;
        chat()/chat_with_tools() are untouched (other callers rely on their returns)."""
        body = {
            "model": settings.llm_model,
            "messages": messages,
            "tools": tools,
            "stream": True,
            "options": {"temperature": settings.llm_temperature},
        }
        tool_calls: list | None = None
        async with self._client.stream(
            "POST",
            f"{settings.ollama_url}/api/chat",
            json=body,
            timeout=settings.llm_timeout_seconds,
        ) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.strip():
                    continue
                data = json.loads(line)
                msg = data.get("message") or {}
                if msg.get("tool_calls"):
                    tool_calls = msg["tool_calls"]
                content = msg.get("content") or ""
                if content:
                    yield ("token", content)
                if data.get("done"):
                    break
        yield ("final", {"role": "assistant", "content": "", "tool_calls": tool_calls})
