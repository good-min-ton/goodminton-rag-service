"""pgvector image-similarity search over product_image_embeddings."""

import asyncpg

from app.core.config import settings


class ImageSearchService:
    # Per-product MIN cosine distance, ranked ascending, over-fetched (H1, H3, H8).
    _SEARCH_SQL = (
        "SELECT product_id, MIN(embedding <=> $1) AS distance "
        "FROM product_image_embeddings "
        "GROUP BY product_id "
        "ORDER BY distance ASC "
        "LIMIT $2"
    )

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def search(
        self, query_embedding: list[float], top_k: int | None = None
    ) -> list[str]:
        k = top_k or settings.image_search_top_k
        over_fetch = k * settings.image_search_over_fetch_factor  # H8
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(self._SEARCH_SQL, query_embedding, over_fetch)
        # product_ids as STRINGS, already ranked by the query (H1).
        return [str(r["product_id"]) for r in rows]
