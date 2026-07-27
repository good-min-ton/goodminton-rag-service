"""pgvector similarity search."""

from dataclasses import dataclass

import asyncpg

from app.core.config import settings


@dataclass
class Chunk:
    doc_type: str
    source_id: str
    chunk_index: int
    content: str
    distance: float


class RetrievalService:
    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def search(
        self,
        query_embedding: list[float],
        k: int | None = None,
        categories: list[str] | None = None,
        doc_type: str | None = None,
    ) -> list[Chunk]:
        """Cosine search in kb_chunks. Optional category filter (metadata->>'category')
        and, for multiple categories, a per-category quota merge so each requested
        category is represented rather than swamped by a nearer one."""
        top_k = k or settings.retrieval_top_k
        if categories and len(categories) > 1:
            per = max(1, top_k // len(categories))
            merged: list[Chunk] = []
            for cat in categories:
                merged.extend(
                    await self._search_one(query_embedding, per, [cat], doc_type)
                )
            return merged
        return await self._search_one(query_embedding, top_k, categories, doc_type)

    async def _search_one(
        self,
        query_embedding: list[float],
        top_k: int,
        categories: list[str] | None,
        doc_type: str | None,
    ) -> list[Chunk]:
        where = []
        params: list = [query_embedding]
        if doc_type:
            params.append(doc_type)
            where.append(f"doc_type = ${len(params)}")
        if categories:
            params.append(categories)
            where.append(f"metadata->>'category' = ANY(${len(params)}::text[])")
        params.append(top_k)
        limit_pos = len(params)
        where_sql = ("WHERE " + " AND ".join(where)) if where else ""
        sql = f"""
            SELECT doc_type, source_id, chunk_index, content,
                   (embedding <=> $1) AS distance
            FROM kb_chunks
            {where_sql}
            ORDER BY embedding <=> $1
            LIMIT ${limit_pos}
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        return [
            Chunk(
                doc_type=row["doc_type"],
                source_id=row["source_id"],
                chunk_index=row["chunk_index"],
                content=row["content"],
                distance=float(row["distance"]),
            )
            for row in rows
        ]
