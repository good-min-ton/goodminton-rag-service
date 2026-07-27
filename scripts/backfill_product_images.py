"""One-time backfill: embed all visible products' images into product_image_embeddings.

Reuses the visible-product id source from backfill_products (SELECT id FROM products
WHERE is_visible = true). Idempotent — re-running just re-embeds.

Run:
    docker compose -f docker-compose.prod.yml exec -T rag-service \\
        uv run python scripts/backfill_product_images.py
"""

import asyncio
import logging
import os
import sys

import asyncpg
import httpx
from pgvector.asyncpg import register_vector

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.embed_client import EmbedClient  # noqa: E402
from app.services.image_indexer import ImageIndexer  # noqa: E402
from app.services.product_client import ProductClient  # noqa: E402
from scripts.backfill_products import (  # noqa: E402
    fetch_product_ids,
    resolve_database_url,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


async def main() -> None:
    pool = await asyncpg.create_pool(
        dsn=resolve_database_url(),
        init=lambda conn: register_vector(conn),
    )

    async with httpx.AsyncClient() as http_client:
        embed = EmbedClient(http_client)
        product_client = ProductClient(http_client)
        indexer = ImageIndexer(pool, embed, product_client, http_client)

        async with pool.acquire() as conn:
            ids = await fetch_product_ids(conn)
        log.info("Found %d visible products to index images for", len(ids))

        succeeded = 0
        failed = 0
        for pid in ids:
            try:
                await indexer.index_product_images(pid)
                succeeded += 1
            except Exception:
                log.exception("Failed to index images for product %s", pid)
                failed += 1

        log.info("Done. Succeeded: %d | Failed: %d", succeeded, failed)

    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
