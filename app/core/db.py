"""Asyncpg connection pool + pgvector type registration."""

import asyncpg
from pgvector.asyncpg import register_vector

from app.core.config import settings


async def _init_connection(conn: asyncpg.Connection) -> None:
    """Register pgvector type cho mỗi connection mới trong pool."""
    await register_vector(conn)


async def create_pool() -> asyncpg.Pool:
    """Tạo asyncpg pool. Gọi trong FastAPI lifespan startup."""
    return await asyncpg.create_pool(
        dsn=settings.resolved_database_url,
        min_size=1,
        max_size=10,
        init=_init_connection,
    )
