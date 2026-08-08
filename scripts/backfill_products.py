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


async def fetch_unindexed_product_ids(conn: asyncpg.Connection) -> list[int]:
    """Visible products with no chunk in kb_chunks — what a partially failed
    backfill leaves behind. Used by bootstrap so a half-populated knowledge base
    heals on the next `up -d` instead of staying half-populated forever."""
    rows = await conn.fetch(
        """
        SELECT p.id
        FROM products p
        WHERE p.is_visible = true
          AND NOT EXISTS (
              SELECT 1 FROM kb_chunks k
              WHERE k.doc_type = 'product' AND k.source_id = p.id::text
          )
        ORDER BY p.id
        """
    )
    return [r["id"] for r in rows]


async def index_ids(ids: list[int]) -> int:
    """Index the given products. Returns the failure count so callers can decide
    whether the run was a success — logging it is not enough, a one-shot
    container that exits 0 reports success no matter how much it dropped."""
    if not ids:
        log.info("Nothing to index")
        return 0

    pool = await asyncpg.create_pool(
        dsn=resolve_database_url(),
        init=lambda conn: register_vector(conn),
    )
    try:
        async with httpx.AsyncClient() as http_client:
            indexer = ProductIndexer(
                pool, EmbeddingService(http_client), ProductClient(http_client)
            )
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
        return failed
    finally:
        await pool.close()


async def main() -> int:
    """Full rebuild of every visible product. Returns the failure count."""
    pool = await asyncpg.create_pool(dsn=resolve_database_url())
    try:
        async with pool.acquire() as conn:
            ids = await fetch_product_ids(conn)
    finally:
        await pool.close()
    log.info("Found %d visible products to index", len(ids))
    return await index_ids(ids)


if __name__ == "__main__":
    # Non-zero on any failure: the rag-init container is the only signal that a
    # cold start actually populated the knowledge base.
    sys.exit(1 if asyncio.run(main()) else 0)
