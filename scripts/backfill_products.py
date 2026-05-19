"""One-time backfill: index all existing products from shop-api into kb_chunks.

Use after first deployment, or whenever product chunks need full rebuild.
Idempotent — re-running just re-embeds.

Run:
    docker compose -f docker-compose.prod.yml exec -T rag-service \\
        uv run python scripts/backfill_products.py
"""

import asyncio
import logging
import os
import sys
from urllib.parse import quote_plus

import asyncpg
import httpx
from pgvector.asyncpg import register_vector

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.embedding import EmbeddingService  # noqa: E402
from app.services.indexer import ProductIndexer  # noqa: E402
from app.services.product_client import ProductClient  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def resolve_database_url() -> str:
    direct = os.getenv("DATABASE_URL")
    if direct:
        return direct
    user = os.getenv("POSTGRES_USER")
    pwd = os.getenv("POSTGRES_PASSWORD")
    if not (user and pwd):
        raise RuntimeError("Need DATABASE_URL or POSTGRES_USER+PASSWORD")
    host = os.getenv("POSTGRES_HOST", "postgres")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "goodminton")
    return f"postgresql://{quote_plus(user)}:{quote_plus(pwd)}@{host}:{port}/{db}"


async def fetch_product_ids(conn: asyncpg.Connection) -> list[int]:
    """Read all product IDs directly from products table (shop-api's DB)."""
    rows = await conn.fetch(
        "SELECT id FROM products WHERE is_visible = true ORDER BY id"
    )
    return [r["id"] for r in rows]


async def main() -> None:
    pool = await asyncpg.create_pool(
        dsn=resolve_database_url(),
        init=lambda conn: register_vector(conn),
    )

    async with httpx.AsyncClient() as http_client:
        embedding = EmbeddingService(http_client)
        product_client = ProductClient(http_client)
        indexer = ProductIndexer(pool, embedding, product_client)

        async with pool.acquire() as conn:
            ids = await fetch_product_ids(conn)
        log.info("Found %d visible products to index", len(ids))

        succeeded = 0
        failed = 0
        for pid in ids:
            try:
                await indexer.index_product(pid)
                succeeded += 1
            except Exception:
                log.exception("Failed to index product %s", pid)
                failed += 1

        log.info("Done. Succeeded: %d | Failed: %d", succeeded, failed)

    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
