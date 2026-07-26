# tests/conftest.py
import os

import httpx
import pytest
import pytest_asyncio

from app.core.db import _init_connection

# DDL mirroring shop-api migration V7__add_rag_vector_store.sql so DB-backed
# tests are self-contained against any empty pgvector Postgres.
_ENSURE_SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS kb_chunks (
    id BIGSERIAL PRIMARY KEY,
    doc_type VARCHAR(20) NOT NULL,
    source_id VARCHAR(200) NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding VECTOR(1024),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_kb_chunk UNIQUE (doc_type, source_id, chunk_index)
);
"""


def _make_embedding(dims: dict[int, float]) -> list[float]:
    """Build a 1024-dim vector: mostly-zero with a few set dims for exact cosine control."""
    vec = [0.0] * 1024
    for i, v in dims.items():
        vec[i] = v
    return vec


@pytest_asyncio.fixture
async def pg_pool():
    """asyncpg pool against a pgvector Postgres.

    Skips (never errors) unless DATABASE_URL is explicitly set. We read ONLY the
    env var — settings.resolved_database_url raises ValueError when creds are
    absent, so it must NOT be evaluated as a fallback here.
    """
    import asyncpg

    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        pytest.skip("DATABASE_URL not set; skipping DB-backed test")
    # Bootstrap the pgvector extension via a short-lived plain connection BEFORE
    # creating the pool. create_pool(..., init=_init_connection) eagerly opens a
    # connection and runs register_vector during pool creation, which fails with
    # "unknown type: public.vector" if the extension isn't created yet. This must
    # use no init=/codec so it works on a virgin DB.
    bootstrap = await asyncpg.connect(dsn)
    try:
        await bootstrap.execute("CREATE EXTENSION IF NOT EXISTS vector")
    finally:
        await bootstrap.close()
    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=4, init=_init_connection)
    try:
        yield pool
    finally:
        await pool.close()


@pytest_asyncio.fixture
async def ensure_kb_schema(pg_pool):
    """Ensure the vector extension + kb_chunks table exist (V7 equivalent)."""
    async with pg_pool.acquire() as conn:
        await conn.execute(_ENSURE_SCHEMA_SQL)
    yield pg_pool


@pytest_asyncio.fixture
async def truncate_kb(ensure_kb_schema):
    pool = ensure_kb_schema
    async with pool.acquire() as conn:
        await conn.execute("TRUNCATE kb_chunks;")
    yield pool


async def seed_chunk(pool, doc_type, source_id, chunk_index, content, embedding):
    """Insert one kb_chunks row; embedding is a python list[float] of length 1024."""
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO kb_chunks (doc_type, source_id, chunk_index, content, embedding) "
            "VALUES ($1, $2, $3, $4, $5)",
            doc_type,
            str(source_id),
            chunk_index,
            content,
            embedding,
        )


@pytest.fixture
def make_embedding():
    return _make_embedding


@pytest_asyncio.fixture
async def similar_client(truncate_kb):
    """ASGI client with ONLY app.state.similar wired to a real DB-backed service.

    Does NOT run the app lifespan, so RabbitMQ (consumer.start) and Ollama are
    never contacted. Depends on truncate_kb -> pg_pool, so it SKIPS cleanly when
    DATABASE_URL is unset. Request truncate_kb alongside it in a test to seed the
    same pool (fixtures are cached per test).
    """
    from app.main import app
    from app.services.similar import SimilarProductsService

    app.state.similar = SimilarProductsService(truncate_kb)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest_asyncio.fixture
async def stub_similar_client():
    """ASGI client with app.state.similar stubbed (no DB, no lifespan).

    For request-validation tests that never reach the service layer, so they run
    everywhere (no DATABASE_URL required).
    """
    from unittest.mock import AsyncMock

    from app.main import app

    app.state.similar = AsyncMock()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
