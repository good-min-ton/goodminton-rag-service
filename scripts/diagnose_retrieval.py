"""Diagnose why a query returns no product context.

Prints chunk counts per doc_type, then the top-k chunks the retriever would
actually feed the LLM for a given query.

Run:
    docker compose -f docker-compose.prod.yml exec -T rag-service \\
        uv run python scripts/diagnose_retrieval.py "vot Astrox 99 gia bao nhieu"
"""

import asyncio
import logging
import os
import sys

import asyncpg
import httpx
from pgvector.asyncpg import register_vector

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.embedding import EmbeddingService  # noqa: E402
from app.services.retrieval import RetrievalService  # noqa: E402
from backfill_products import resolve_database_url  # noqa: E402

logging.basicConfig(level=logging.WARNING)


async def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "vot cau long cho nguoi moi choi"

    pool = await asyncpg.create_pool(
        dsn=resolve_database_url(), init=lambda conn: register_vector(conn)
    )

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT doc_type,
                   COUNT(*) AS chunks,
                   COUNT(DISTINCT source_id) AS sources
            FROM kb_chunks GROUP BY doc_type ORDER BY doc_type
            """
        )
        products = await conn.fetchval(
            "SELECT COUNT(*) FROM products WHERE is_visible = true"
        )

    print("kb_chunks:")
    for r in rows:
        print(f"  {r['doc_type']:<8} chunks={r['chunks']:<6} sources={r['sources']}")
    if not rows:
        print("  (empty)")
    print(f"visible products in shop DB: {products}\n")

    async with httpx.AsyncClient() as http_client:
        vec = await EmbeddingService(http_client).embed(query)
        chunks = await RetrievalService(pool).search(vec)

    print(f'top-{len(chunks)} for query: "{query}"')
    for c in chunks:
        preview = " ".join(c.content.split())[:70]
        print(
            f"  dist={c.distance:.3f} {c.doc_type:<8} src={c.source_id:<12} {preview}"
        )

    product_hits = [c for c in chunks if c.doc_type == "product"]
    print(
        f"\nproduct chunks in context: {len(product_hits)} "
        f"-> product_ids passed to LLM: "
        f"{sorted({c.source_id for c in product_hits}) or 'NONE (bot cannot call tools)'}"
    )

    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
