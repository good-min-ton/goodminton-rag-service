"""HTTP client for shop-api internal product endpoint."""

import httpx

from app.core.config import settings


class ProductClient:
    def __init__(self, client: httpx.AsyncClient):
        self._client = client

    async def get_for_rag(self, product_id: int) -> dict:
        r = await self._client.get(
            f"{settings.shop_api_url}/api/internal/products/{product_id}",
            headers={"X-Internal-Key": settings.internal_api_key or ""},
            timeout=10.0,
        )
        r.raise_for_status()
        return r.json()
