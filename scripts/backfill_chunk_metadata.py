"""Re-index all product chunks so kb_chunks.metadata->>'category' is populated.

Run once after deploying the metadata-writing indexer:
    uv run python -m scripts.backfill_chunk_metadata
"""

import asyncio
import logging

import httpx

from app.core.db import create_pool
from app.services.embedding import EmbeddingService
from app.services.indexer import ProductIndexer
from app.services.product_client import ProductClient

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("backfill_chunk_metadata")


async def main() -> None:
    pool = await create_pool()
    http = httpx.AsyncClient()
    indexer = ProductIndexer(pool, EmbeddingService(http), ProductClient(http))
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT DISTINCT source_id FROM kb_chunks WHERE doc_type='product'"
        )
    ids = sorted(int(r["source_id"]) for r in rows)
    log.info("re-indexing %d products", len(ids))
    for pid in ids:
        try:
            await indexer.index_product(pid)
        except Exception as exc:  # keep going; log the failures
            log.warning("product %s failed: %s", pid, exc)
    await http.aclose()
    await pool.close()
    log.info("done")


if __name__ == "__main__":
    asyncio.run(main())
