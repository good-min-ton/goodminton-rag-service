"""Ollama chat wrapper. Phase 3: non-streaming, no tool calling."""

import httpx

from app.core.config import settings


class LLMService:
    def __init__(self, client: httpx.AsyncClient):
        self._client = client

    async def chat(self, messages: list[dict]) -> str:
        """Gửi messages cho Ollama /api/chat, trả về assistant text."""
        r = await self._client.post(
            f"{settings.ollama_url}/api/chat",
            json={
                "model": settings.llm_model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": settings.llm_temperature},
            },
            timeout=settings.llm_timeout_seconds,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]
