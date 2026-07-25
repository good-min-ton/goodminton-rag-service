# app/services/similar.py
from __future__ import annotations

from dataclasses import dataclass

import asyncpg

_NAME_PREFIX = "Sản phẩm: "


class ProductNotIndexedError(Exception):
    """Raised when the source product has no indexed chunks (truly not found).

    Distinct from find_similar returning [] (the product exists but has no
    similar peers), so the router can map only THIS to HTTP 404.
    """


@dataclass
class SimilarResult:
    product_id: str
    name: str | None
    distance: float
    chunk_count: int


def _parse_name(content: str) -> str | None:
    """Best-effort: first line after the 'Sản phẩm: ' prefix in chunk_index=0 text."""
    if not content:
        return None
    first_line = content.split("\n", 1)[0].strip()
    if first_line.startswith(_NAME_PREFIX):
        name = first_line[len(_NAME_PREFIX) :].strip()
        return name or None
    return None


class SimilarProductsService:
    _COUNT_SQL = (
        "SELECT count(*) AS n FROM kb_chunks "
        "WHERE doc_type = 'product' AND source_id = $1"
    )

    _RANK_SQL = (
        "WITH src AS ("
        "  SELECT avg(embedding) AS centroid FROM kb_chunks"
        "  WHERE doc_type = 'product' AND source_id = $1"
        ") "
        "SELECT k.source_id,"
        "       count(*)::int AS chunk_count,"
        "       (avg(k.embedding) <=> (SELECT centroid FROM src)) AS distance "
        "FROM kb_chunks k "
        "WHERE k.doc_type = 'product' AND k.source_id <> $1 "
        "GROUP BY k.source_id "
        "ORDER BY distance ASC "
        "LIMIT $2"
    )

    _NAME_SQL = (
        "SELECT source_id, content FROM kb_chunks "
        "WHERE doc_type = 'product' AND chunk_index = 0 "
        "AND source_id = ANY($1::text[])"
    )

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def find_similar(self, product_id: int, limit: int) -> list[SimilarResult]:
        source_id = str(product_id)
        async with self._pool.acquire() as conn:
            # Existence guard: the source itself must be indexed. Zero chunks ->
            # truly not found (router maps ONLY this to 404).
            count_row = await conn.fetchrow(self._COUNT_SQL, source_id)
            if count_row is None or count_row["n"] == 0:
                raise ProductNotIndexedError(source_id)

            # Centroid-to-centroid cosine ranking, one row per candidate product.
            rows = await conn.fetch(self._RANK_SQL, source_id, limit)
            if not rows:
                # Source is indexed but there are no other products -> empty (200).
                return []

            ids = [r["source_id"] for r in rows]
            # Best-effort name enrichment from chunk_index=0 content.
            name_rows = await conn.fetch(self._NAME_SQL, ids)

        names = {r["source_id"]: _parse_name(r["content"]) for r in name_rows}
        return [
            SimilarResult(
                product_id=r["source_id"],
                name=names.get(r["source_id"]),
                distance=float(r["distance"]),
                chunk_count=int(r["chunk_count"]),
            )
            for r in rows
        ]
