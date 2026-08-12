"""pgvector image-similarity search over product_image_embeddings."""

import asyncpg

from app.core.config import settings


class ImageSearchService:
    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def search(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
        max_distance: float | None = None,
    ) -> list[str]:
        k = top_k or settings.image_search_top_k
        over_fetch = k * settings.image_search_over_fetch_factor  # H8
        params: list = [query_embedding]
        having = ""
        max_dist = (
            settings.image_search_max_distance if max_distance is None else max_distance
        )
        if max_dist and max_dist > 0:
            params.append(max_dist)
            having = f"HAVING MIN(embedding <=> $1) <= ${len(params)} "
        params.append(over_fetch)
        sql = (
            "SELECT product_id, MIN(embedding <=> $1) AS distance "
            "FROM product_image_embeddings "
            "GROUP BY product_id "
            f"{having}"
            "ORDER BY distance ASC "
            f"LIMIT ${len(params)}"
        )
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        # product_ids as STRINGS, already ranked by the query (H1).
        return [str(r["product_id"]) for r in rows]
