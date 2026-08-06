"""HTTP client for the embed-service /rerank endpoint (mirrors app/services/embed_client.py)."""

import httpx

from app.core.config import settings


class RerankUnavailable(Exception):
    """rerank endpoint unreachable or non-200."""


class RerankClient:
    def __init__(self, client: httpx.AsyncClient):
        self._client = client

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        try:
            r = await self._client.post(
                f"{settings.embed_service_url}/rerank",
                json={"query": query, "documents": documents},
                timeout=60.0,
            )
        except httpx.HTTPError as exc:
            raise RerankUnavailable(str(exc)) from exc
        if r.status_code != 200:
            raise RerankUnavailable(f"rerank returned {r.status_code}")
        data = r.json()
        if "scores" not in data or len(data["scores"]) != len(documents):
            raise RerankUnavailable(f"malformed rerank response: {data!r}")
        return data["scores"]
