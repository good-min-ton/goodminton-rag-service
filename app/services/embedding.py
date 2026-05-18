"""Ollama embedding wrapper."""

import httpx

from app.core.config import settings


class EmbeddingService:
    def __init__(self, client: httpx.AsyncClient):
        self._client = client

    async def embed(self, text: str) -> list[float]:
        r = await self._client.post(
            f"{settings.ollama_url}/api/embeddings",
            json={"model": settings.embedding_model, "prompt": text},
            timeout=30.0,
        )
        r.raise_for_status()
        return r.json()["embedding"]
