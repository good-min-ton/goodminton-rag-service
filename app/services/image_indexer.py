# app/services/image_indexer.py
"""Image indexer — downloads a product's images, embeds them, atomically replaces rows.

Read-only against the shop-api catalog: only reads image URLs; writes ONLY
product_image_embeddings.
"""

import logging

import asyncpg
import httpx

from app.core.config import settings
from app.services.embed_client import EmbedClient
from app.services.product_client import ProductClient

log = logging.getLogger(__name__)


class ImageIndexer:
    def __init__(
        self,
        pool: asyncpg.Pool,
        embed_client: EmbedClient,
        product_client: ProductClient,
        http_client: httpx.AsyncClient,
    ):
        self._pool = pool
        self._embed = embed_client
        self._client = product_client
        self._http = http_client

    async def _download(self, url: str) -> bytes:
        r = await self._http.get(url, timeout=30.0)
        r.raise_for_status()
        data = r.content
        if len(data) > settings.image_max_upload_bytes:  # cap decoded size (H9)
            raise ValueError(f"image exceeds byte cap: {url}")
        return data

    async def index_product_images(self, product_id: int) -> int:
        """Embed every image; atomic replace over successes. Returns embedded count.

        Per-image failures are skipped (H9). If ALL fail, existing rows are kept
        (no wipe) and 0 is returned.
        """
        images = await self._client.get_product_images(product_id)
        embedded: list[tuple[int, str, list[float]]] = []
        for img in images:
            resource_id = img["resourceId"]
            url = img["url"]
            try:
                data = await self._download(url)
                embedding = await self._embed.embed_image(data)
            except Exception:
                log.exception(
                    "Failed to embed image %s for product %s", url, product_id
                )
                continue
            embedded.append((resource_id, url, embedding))

        if not embedded:  # H9: never wipe existing rows on total failure
            log.warning(
                "No images embedded for product %s — keeping existing rows", product_id
            )
            return 0

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "DELETE FROM product_image_embeddings WHERE product_id = $1",
                    product_id,
                )
                for resource_id, url, embedding in embedded:
                    await conn.execute(
                        "INSERT INTO product_image_embeddings "
                        "(resource_id, product_id, url, embedding) "
                        "VALUES ($1, $2, $3, $4)",
                        resource_id,
                        product_id,
                        url,
                        embedding,
                    )

        log.info("Indexed %d images for product %s", len(embedded), product_id)
        return len(embedded)

    async def delete_product_images(self, product_id: int) -> int:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM product_image_embeddings WHERE product_id = $1",
                product_id,
            )
        log.info("Deleted image embeddings for product %s (%s)", product_id, result)
        try:
            return int(result.split()[-1])  # asyncpg returns "DELETE N"
        except (ValueError, IndexError):
            return 0
