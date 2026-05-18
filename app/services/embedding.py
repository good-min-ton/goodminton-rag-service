"""Ollama embedding wrapper."""

import httpx

from app.core.config import settings


class EmbeddingService:
    def __init__(self, client: httpx.AsyncClient):
        self._client = client

    async def embed(self, text: str) -> list[float]:
        # Dùng /api/embed (new endpoint, batch-capable) — /api/embeddings đã deprecated
        r = await self._client.post(
            f"{settings.ollama_url}/api/embed",
            json={"model": settings.embedding_model, "input": text},
            timeout=30.0,
        )
        r.raise_for_status()
        return r.json()["embeddings"][0]
