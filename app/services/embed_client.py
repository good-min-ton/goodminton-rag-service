"""HTTP client for the SigLIP embed-service (mirrors product_client.py)."""

import httpx

from app.core.config import settings


class EmbedUnavailable(Exception):
    """embed-service unreachable or returned non-200 — router maps this to HTTP 503 (H7)."""


class EmbedClient:
    def __init__(self, client: httpx.AsyncClient):
        self._client = client

    async def embed_image(self, data: bytes) -> list[float]:
        """POST image bytes as multipart field 'file' (H2); return the 768-dim vector.

        The embed-service gates on a Content-Type starting with 'image/' (then PIL
        sniffs the real format), so declare a generic image type — sending
        'application/octet-stream' is rejected with 400 even for valid images.
        """
        try:
            r = await self._client.post(
                f"{settings.embed_service_url}/embed/image",
                files={"file": ("upload", data, "image/jpeg")},
                timeout=30.0,
            )
        except httpx.HTTPError as exc:  # connect/timeout/etc -> unavailable (H7)
            raise EmbedUnavailable(str(exc)) from exc
        if r.status_code != 200:
            raise EmbedUnavailable(f"embed-service returned {r.status_code}")
        return r.json()["embedding"]
