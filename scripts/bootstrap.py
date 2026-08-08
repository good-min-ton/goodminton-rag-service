"""One-shot bootstrap for fresh deployments (e.g. after a volume wipe).

Idempotent — safe to run on every compose up:
- static chunks missing  -> index static docs
- product chunks missing -> backfill products from shop-api

Run by the `rag-init` service in docker-compose.prod.yml, or manually:
    docker compose -f docker-compose.prod.yml exec -T rag-service \\
        uv run python scripts/bootstrap.py
"""

import asyncio
import logging
import os
import sys

import asyncpg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backfill_products  # noqa: E402
import index_static_docs  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bootstrap")

DB_CONNECT_RETRIES = 30
DB_RETRY_DELAY_SECONDS = 2


async def wait_for_db(dsn: str) -> asyncpg.Connection:
    last_error: Exception | None = None
    for attempt in range(1, DB_CONNECT_RETRIES + 1):
        try:
            return await asyncpg.connect(dsn)
        except Exception as e:
            last_error = e
            log.info("DB not ready (attempt %d/%d)", attempt, DB_CONNECT_RETRIES)
            await asyncio.sleep(DB_RETRY_DELAY_SECONDS)
    raise RuntimeError(f"Database unreachable: {last_error}")


async def main() -> int:
    """Returns the number of products that failed to index."""
    dsn = backfill_products.resolve_database_url()
    conn = await wait_for_db(dsn)
    try:
        static_count = await conn.fetchval(
            "SELECT COUNT(*) FROM kb_chunks WHERE doc_type = 'static'"
        )
        indexed = await conn.fetchval(
            "SELECT COUNT(DISTINCT source_id) FROM kb_chunks WHERE doc_type = 'product'"
        )
        # Compare against the source of truth, not against zero. The old guard was
        # `product_count == 0`, so a backfill that died halfway (Ollama restarting,
        # shop-api not up yet) left chunks behind and every later run reported
        # "present -> skip". The missing products were then never indexed, and a
        # question about one of them retrieved whatever else was in the store.
        missing = await backfill_products.fetch_unindexed_product_ids(conn)
    finally:
        await conn.close()

    log.info(
        "kb_chunks: static=%d, products indexed=%d, missing=%d",
        static_count,
        indexed,
        len(missing),
    )

    if static_count == 0:
        log.info("Static chunks missing -> indexing static docs")
        await index_static_docs.main()
    else:
        log.info("Static chunks present -> skip")

    if missing:
        log.info("Indexing %d product(s) with no chunks", len(missing))
        failed = await backfill_products.index_ids(missing)
    else:
        log.info("Every visible product is indexed -> skip")
        failed = 0

    log.info("Bootstrap complete")
    return failed


if __name__ == "__main__":
    # Non-zero when products were left unindexed, so a cold start that only
    # half-populated the knowledge base shows up as a failed rag-init container
    # instead of passing quietly.
    sys.exit(1 if asyncio.run(main()) else 0)
