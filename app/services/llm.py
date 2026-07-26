"""Ollama chat wrapper. Phase 3: non-streaming, no tool calling."""

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
