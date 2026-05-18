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

    async def search(self, query_embedding: list[float], k: int | None = None) -> list[Chunk]:
        """Cosine similarity search trong kb_chunks. Trả về top-k chunks."""
        top_k = k or settings.retrieval_top_k

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT doc_type, source_id, chunk_index, content,
                       (embedding <=> $1) AS distance
                FROM kb_chunks
                ORDER BY embedding <=> $1
                LIMIT $2
                """,
                query_embedding,
                top_k,
            )

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
