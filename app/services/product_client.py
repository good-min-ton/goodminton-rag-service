"""HTTP client for shop-api internal endpoints."""

import httpx

from app.core.config import settings


class ProductClient:
    def __init__(self, client: httpx.AsyncClient):
        self._client = client

    def _headers(self) -> dict[str, str]:
        return {"X-Internal-Key": settings.internal_api_key or ""}

    async def get_for_rag(self, product_id: int) -> dict:
        r = await self._client.get(
            f"{settings.shop_api_url}/api/internal/products/{product_id}",
            headers=self._headers(),
            timeout=10.0,
        )
        r.raise_for_status()
        return r.json()

    async def get_pricing(self, product_id: int) -> dict:
        r = await self._client.get(
            f"{settings.shop_api_url}/api/internal/products/{product_id}/pricing",
            headers=self._headers(),
            timeout=10.0,
        )
        r.raise_for_status()
        return r.json()

    async def check_inventory(self, variant_id: int) -> list[dict]:
        r = await self._client.get(
            f"{settings.shop_api_url}/api/internal/variants/{variant_id}/inventory",
            headers=self._headers(),
            timeout=10.0,
        )
        r.raise_for_status()
        return r.json()
