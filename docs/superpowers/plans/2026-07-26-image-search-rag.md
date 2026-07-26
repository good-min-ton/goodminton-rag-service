# Image Search (RAG) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add image-similarity search to the RAG service — embed an uploaded image via a separate embed-service, rank products by pgvector cosine distance, and keep `product_image_embeddings` fresh via an indexer, RabbitMQ hook, and backfill script.

**Architecture:** A new `EmbedClient` (mirroring `product_client.py`) POSTs image bytes to the embed-service (SigLIP, port 8001) and returns a 768-dim vector. `ImageSearchService` runs `MIN(embedding <=> $1) GROUP BY product_id` over `product_image_embeddings` and returns ranked product-id **strings**. A `POST /search/image` router validates + rate-limits the upload, embeds it, searches, and returns `{"product_ids": [...]}`. `ImageIndexer` (fed by the existing product consumer + a new backfill script) downloads each product's Cloudinary images from shop-api, embeds them, and atomically replaces that product's rows.

**Tech Stack:** FastAPI, asyncpg + pgvector, httpx (multipart + MockTransport in tests), pytest / pytest-asyncio, ruff.

## Global Constraints

Every task inherits these — copied verbatim from the spec (§ Global Constraints & Decisions and Component Design §2):

- **H1 — one response contract:** `POST /search/image` returns `{"product_ids": ["42","17",…]}` — **strings, ranked by ascending distance**. No `results` / object-array drift.
- **H2 — one multipart field name `file`** across all hops: FE→RAG uses `file`; RAG→embed-service uses `file`.
- **H3 — no distance threshold initially:** return top-K regardless of distance. No `WHERE distance < …` filter.
- **H5 — rate-limit** `/search/image` (unauthenticated + CPU-bound): a crude per-IP limit in RAG.
- **H7 — graceful 503:** if the embed-service is unreachable / returns non-200, `/search/image` returns a friendly **503** ("Tìm ảnh đang khởi động, thử lại"), never a 500 stacktrace. RAG `/health` does **not** depend on the embed-service.
- **H8 — over-fetch:** `ImageSearchService` fetches `$2 = top_k * over_fetch_factor` (factor = 3) ids so hydration can keep the first `top_k` visible ones downstream.
- **H9 — resilient indexing:** per-image download+embed wrapped in try/except; skip failures (log); atomic replace over whatever succeeded; **if ALL images fail, do NOT wipe existing rows** (skip the replace). Cap decoded byte size on this path too.
- **H10 — score sanity:** the response carries only `product_ids` (no score). If a score is ever surfaced, clamp `similarity = max(0, 1 - distance)`.
- **Note — `app/main.py` line numbers drift:** Task 4 inserts several lines into `app/main.py`, so the `:NN` citations in Tasks 4–6 are relative to the PRE-EDIT file. When editing `main.py`, anchor on the quoted surrounding code (e.g. right after `indexer = ProductIndexer(...)` or the `app.state.description = …` block), NOT the raw line number, and re-read the file after each insert.
- **Read-only to the shop-api catalog:** RAG never writes `products`/`resources`; it only reads image URLs via `GET /api/internal/products/{id}/images` (X-Internal-Key) and writes ONLY `product_image_embeddings`.
- **The vector table is `product_image_embeddings(resource_id PK, product_id, url, embedding vector(768))`** — SigLIP `google/siglip-base-patch16-224`, 768-dim, L2-normalized. (shop-api owns the real migration; tests create an equivalent table.)

Verification (whole suite, run from repo root):
`DATABASE_URL=postgresql://postgres:postgres@localhost:5432/goodminton_test uv run pytest`
then `uv run ruff format --check .` and `uv run ruff check .`
(CI runs `ruff format --check` — always run `uv run ruff format .` before committing.)

Non-DB tests (embed client, search router, consumer, backfill) do not need `DATABASE_URL`; the DB-backed tests (image search service, image indexer) skip cleanly when it is unset.

---

### Task 1: Config — image-search settings

**Files:**
- Modify: `app/core/config.py:59-63` (append after the "Shop API internal endpoint" block, before the `@property` at line 65)
- Test: `tests/test_config_image_search.py` (create)

**Interfaces:**
- Consumes: nothing
- Produces: `settings.embed_service_url: str`, `settings.image_search_top_k: int`, `settings.image_search_over_fetch_factor: int`, `settings.image_max_upload_bytes: int`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_image_search.py
def test_image_search_settings_defaults():
    from app.core.config import settings

    assert settings.embed_service_url == "http://localhost:8001"
    assert settings.image_search_top_k > 0
    assert settings.image_search_over_fetch_factor == 3
    assert settings.image_max_upload_bytes == 8 * 1024 * 1024
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config_image_search.py -v`
Expected: FAIL with `AttributeError: 'Settings' object has no attribute 'embed_service_url'`

- [ ] **Step 3: Write minimal implementation**

Insert after line 63 (`central_store_name = ...`) in `app/core/config.py`:

```python

    # Image search — embed-service (SigLIP, port 8001) + pgvector
    embed_service_url: str = "http://localhost:8001"
    image_search_top_k: int = 12
    image_search_over_fetch_factor: int = 3  # over-fetch = top_k * this (H8)
    image_max_upload_bytes: int = 8 * 1024 * 1024  # 8 MB outer cap
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config_image_search.py -v`
Expected: PASS

- [ ] **Step 5: Format, lint, commit**

```bash
uv run ruff format . && uv run ruff check .
git add app/core/config.py tests/test_config_image_search.py
git commit -m "feat(config): add image-search settings (embed_service_url, top_k, over_fetch, byte cap)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: EmbedClient + EmbedUnavailable

**Files:**
- Create: `app/services/embed_client.py`
- Test: `tests/test_embed_client.py` (create)

Mirrors `app/services/product_client.py:8-22` (takes a shared `httpx.AsyncClient`, reads its base URL from `settings`).

**Interfaces:**
- Consumes: `settings.embed_service_url` (Task 1)
- Produces: `class EmbedUnavailable(Exception)`; `class EmbedClient: def __init__(self, client: httpx.AsyncClient); async def embed_image(self, data: bytes) -> list[float]` — POSTs multipart field `file` (H2) to `{embed_service_url}/embed/image`, returns the `embedding` list; raises `EmbedUnavailable` on connect error / non-200 (H7).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_embed_client.py
import httpx
import pytest

from app.services.embed_client import EmbedClient, EmbedUnavailable


@pytest.mark.asyncio
async def test_embed_image_returns_vector_and_posts_file_field():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["content_type"] = request.headers.get("content-type", "")
        captured["body"] = request.content
        return httpx.Response(200, json={"embedding": [0.1] * 768})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        vec = await EmbedClient(client).embed_image(b"rawbytes")

    assert len(vec) == 768
    assert captured["content_type"].startswith("multipart/form-data")
    assert b'name="file"' in captured["body"]  # H2: multipart field 'file'


@pytest.mark.asyncio
async def test_embed_image_non_200_raises_embed_unavailable():
    transport = httpx.MockTransport(lambda req: httpx.Response(500, text="boom"))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(EmbedUnavailable):
            await EmbedClient(client).embed_image(b"x")


@pytest.mark.asyncio
async def test_embed_image_connect_error_raises_embed_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(EmbedUnavailable):
            await EmbedClient(client).embed_image(b"x")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_embed_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.embed_client'`

- [ ] **Step 3: Write minimal implementation**

```python
# app/services/embed_client.py
"""HTTP client for the SigLIP embed-service (mirrors product_client.py)."""

import httpx

from app.core.config import settings


class EmbedUnavailable(Exception):
    """embed-service unreachable or returned non-200 — router maps this to HTTP 503 (H7)."""


class EmbedClient:
    def __init__(self, client: httpx.AsyncClient):
        self._client = client

    async def embed_image(self, data: bytes) -> list[float]:
        """POST image bytes as multipart field 'file' (H2); return the 768-dim vector."""
        try:
            r = await self._client.post(
                f"{settings.embed_service_url}/embed/image",
                files={"file": ("upload", data, "application/octet-stream")},
                timeout=30.0,
            )
        except httpx.HTTPError as exc:  # connect/timeout/etc -> unavailable (H7)
            raise EmbedUnavailable(str(exc)) from exc
        if r.status_code != 200:
            raise EmbedUnavailable(f"embed-service returned {r.status_code}")
        return r.json()["embedding"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_embed_client.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Format, lint, commit**

```bash
uv run ruff format . && uv run ruff check .
git add app/services/embed_client.py tests/test_embed_client.py
git commit -m "feat(embed): add EmbedClient posting multipart 'file' + typed EmbedUnavailable (H2/H7)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: ImageSearchService (pgvector ranking)

**Files:**
- Create: `app/services/image_search.py`
- Modify: `tests/conftest.py:26` (add image-embeddings schema/fixtures/seed helper after the `_ENSURE_SCHEMA_SQL` block and its fixtures)
- Test: `tests/test_image_search_service.py` (create)

SQL pattern mirrors `app/services/similar.py:44-57` (asyncpg `GROUP BY` + `<=>` cosine + `ORDER BY distance ASC LIMIT`).

**Interfaces:**
- Consumes: `settings.image_search_top_k`, `settings.image_search_over_fetch_factor` (Task 1); an `asyncpg.Pool`
- Produces: `class ImageSearchService: def __init__(self, pool: asyncpg.Pool); async def search(self, query_embedding: list[float], top_k: int | None = None) -> list[str]` — returns up to `top_k * over_fetch_factor` product-id **strings** ranked by ascending per-product MIN distance (H1, H3, H8).
- Produces (conftest): `truncate_image_embeddings` fixture (yields the pool with an empty `product_image_embeddings` table), `make_embedding_768` fixture, `seed_image_embedding(pool, resource_id, product_id, url, embedding)` coroutine.

- [ ] **Step 1: Write the failing test**

First add the fixtures to `tests/conftest.py`. Insert after the `truncate_kb` fixture / `seed_chunk` helper block (i.e. after line 94, before the `make_embedding` fixture at line 97):

```python
# --- Image search (product_image_embeddings, vector(768)) ---
_ENSURE_IMAGE_SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS product_image_embeddings (
    resource_id INTEGER PRIMARY KEY,
    product_id INTEGER NOT NULL,
    url TEXT,
    embedding VECTOR(768)
);
"""


def _make_embedding_768(dims: dict[int, float]) -> list[float]:
    """Build a 768-dim vector: mostly-zero with a few set dims for exact cosine control."""
    vec = [0.0] * 768
    for i, v in dims.items():
        vec[i] = v
    return vec


@pytest_asyncio.fixture
async def truncate_image_embeddings(pg_pool):
    async with pg_pool.acquire() as conn:
        await conn.execute(_ENSURE_IMAGE_SCHEMA_SQL)
        await conn.execute("TRUNCATE product_image_embeddings;")
    yield pg_pool


@pytest.fixture
def make_embedding_768():
    return _make_embedding_768


async def seed_image_embedding(pool, resource_id, product_id, url, embedding):
    """Insert one product_image_embeddings row; embedding is a list[float] len 768."""
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO product_image_embeddings "
            "(resource_id, product_id, url, embedding) VALUES ($1, $2, $3, $4)",
            resource_id,
            product_id,
            url,
            embedding,
        )
```

Then the service test:

```python
# tests/test_image_search_service.py
import pytest

from app.services.image_search import ImageSearchService


@pytest.mark.asyncio
async def test_search_ranks_by_per_product_min_distance(
    truncate_image_embeddings, make_embedding_768
):
    from tests.conftest import seed_image_embedding

    pool = truncate_image_embeddings
    # product 1: a near image (dim0) AND a far image (dim5) -> MIN keeps it nearest.
    await seed_image_embedding(pool, 1, 1, "a", make_embedding_768({0: 1.0}))
    await seed_image_embedding(pool, 2, 1, "b", make_embedding_768({5: 1.0}))
    # product 2: moderately close.
    await seed_image_embedding(pool, 3, 2, "c", make_embedding_768({0: 0.9, 1: 0.1}))
    # product 3: far.
    await seed_image_embedding(pool, 4, 3, "d", make_embedding_768({5: 1.0}))

    svc = ImageSearchService(pool)
    ids = await svc.search(make_embedding_768({0: 1.0}), top_k=10)

    # Deduped per product, ranked by ascending MIN distance, ids are STRINGS (H1).
    assert ids == ["1", "2", "3"]


@pytest.mark.asyncio
async def test_search_over_fetches_top_k_times_factor(
    truncate_image_embeddings, make_embedding_768, monkeypatch
):
    from app.core.config import settings
    from tests.conftest import seed_image_embedding

    pool = truncate_image_embeddings
    # 5 products, 1 image each, distance increases with pid.
    for pid in range(1, 6):
        await seed_image_embedding(
            pool, pid, pid, f"u{pid}", make_embedding_768({0: 1.0, 700: 0.05 * pid})
        )

    monkeypatch.setattr(settings, "image_search_over_fetch_factor", 3)
    svc = ImageSearchService(pool)
    ids = await svc.search(make_embedding_768({0: 1.0}), top_k=1)

    # top_k=1 but over_fetch = 1*3 = 3 -> returns 3 candidates, not 1 (H8).
    assert ids == ["1", "2", "3"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `DATABASE_URL=postgresql://postgres:postgres@localhost:5432/goodminton_test uv run pytest tests/test_image_search_service.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.image_search'`

- [ ] **Step 3: Write minimal implementation**

```python
# app/services/image_search.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `DATABASE_URL=postgresql://postgres:postgres@localhost:5432/goodminton_test uv run pytest tests/test_image_search_service.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Format, lint, commit**

```bash
uv run ruff format . && uv run ruff check .
git add app/services/image_search.py tests/conftest.py tests/test_image_search_service.py
git commit -m "feat(search): add ImageSearchService (per-product MIN cosine, over-fetch, string ids)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: POST /search/image router + rate limit + main.py wiring

**Files:**
- Create: `app/routers/search.py`
- Modify: `app/main.py:16` (add `from app.routers import search as search_router`), `:21` (add EmbedClient + ImageSearchService imports), `:38` (create `embed_client`), `:55` (add `app.state.embed` + `app.state.image_search`), `:83` (add `app.include_router(search_router.router)`)
- Modify: `pyproject.toml` (add `python-multipart` — required by FastAPI to parse multipart uploads)
- Test: `tests/test_search_router.py` (create)

Router pattern mirrors `app/routers/products.py:23-52` (reads services off `http_request.app.state`, raises `HTTPException`). App-state stubbing in tests mirrors `tests/conftest.py:120-134` (`stub_similar_client`) — ASGITransport without running lifespan.

**Interfaces:**
- Consumes: `EmbedClient.embed_image(bytes) -> list[float]` + `EmbedUnavailable` (Task 2); `ImageSearchService.search(list[float], top_k=None) -> list[str]` (Task 3); `settings.image_max_upload_bytes` (Task 1)
- Produces: `router = APIRouter(prefix="/search", tags=["search"])` with `POST /search/image` returning `{"product_ids": [...]}` (H1); module-level `_hits` dict + `_RATE_MAX` int (per-IP limiter, H5); `app.state.embed` and `app.state.image_search` wired in `main.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_search_router.py
from unittest.mock import AsyncMock

import httpx
import pytest

from app.main import app
from app.services.embed_client import EmbedUnavailable


def _client() -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_search_image_returns_product_ids():
    app.state.embed = AsyncMock()
    app.state.embed.embed_image.return_value = [0.1] * 768
    app.state.image_search = AsyncMock()
    app.state.image_search.search.return_value = ["42", "17"]

    async with _client() as c:
        r = await c.post(
            "/search/image", files={"file": ("q.jpg", b"imgbytes", "image/jpeg")}
        )

    assert r.status_code == 200
    assert r.json() == {"product_ids": ["42", "17"]}  # H1


@pytest.mark.asyncio
async def test_search_image_embed_unavailable_returns_503():
    app.state.embed = AsyncMock()
    app.state.embed.embed_image.side_effect = EmbedUnavailable("down")
    app.state.image_search = AsyncMock()

    async with _client() as c:
        r = await c.post(
            "/search/image", files={"file": ("q.jpg", b"imgbytes", "image/jpeg")}
        )

    assert r.status_code == 503  # H7


@pytest.mark.asyncio
async def test_search_image_rejects_non_image_content_type():
    app.state.embed = AsyncMock()
    app.state.image_search = AsyncMock()

    async with _client() as c:
        r = await c.post(
            "/search/image", files={"file": ("q.txt", b"hello", "text/plain")}
        )

    assert r.status_code == 400


@pytest.mark.asyncio
async def test_search_image_rate_limited_returns_429(monkeypatch):
    from app.routers import search as search_mod

    search_mod._hits.clear()
    monkeypatch.setattr(search_mod, "_RATE_MAX", 1)
    app.state.embed = AsyncMock()
    app.state.embed.embed_image.return_value = [0.1] * 768
    app.state.image_search = AsyncMock()
    app.state.image_search.search.return_value = []

    async with _client() as c:
        r1 = await c.post(
            "/search/image", files={"file": ("q.jpg", b"imgbytes", "image/jpeg")}
        )
        r2 = await c.post(
            "/search/image", files={"file": ("q.jpg", b"imgbytes", "image/jpeg")}
        )

    assert r1.status_code == 200
    assert r2.status_code == 429  # H5
    search_mod._hits.clear()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_search_router.py -v`
Expected: FAIL — `404 Not Found` on `/search/image` (route not registered) / `ModuleNotFoundError: app.routers.search`.

- [ ] **Step 3a: Add the multipart dependency**

Run: `uv add python-multipart`
(FastAPI raises `RuntimeError: Form data requires "python-multipart"` when parsing `UploadFile` without it.)

- [ ] **Step 3b: Write the router**

```python
# app/routers/search.py
"""POST /search/image — visual product search (embed -> pgvector -> product_ids)."""

import time
from collections import defaultdict, deque

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.core.config import settings
from app.services.embed_client import EmbedClient, EmbedUnavailable
from app.services.image_search import ImageSearchService

router = APIRouter(prefix="/search", tags=["search"])

# Crude in-memory per-IP rate limit (H5). Unauthenticated + CPU-bound endpoint.
_RATE_WINDOW_SECONDS = 60
_RATE_MAX = 20
_hits: dict[str, deque[float]] = defaultdict(deque)


def _rate_limited(ip: str) -> bool:
    now = time.monotonic()
    dq = _hits[ip]
    while dq and now - dq[0] > _RATE_WINDOW_SECONDS:
        dq.popleft()
    if len(dq) >= _RATE_MAX:
        return True
    dq.append(now)
    return False


@router.post("/image")
async def search_image(
    http_request: Request, file: UploadFile = File(...)
) -> dict:
    ip = http_request.client.host if http_request.client else "unknown"
    if _rate_limited(ip):
        raise HTTPException(status_code=429, detail="Quá nhiều yêu cầu, thử lại sau")

    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="File phải là ảnh")

    data = await file.read()
    if len(data) > settings.image_max_upload_bytes:
        raise HTTPException(status_code=413, detail="Ảnh quá lớn")

    embed: EmbedClient = http_request.app.state.embed
    search_svc: ImageSearchService = http_request.app.state.image_search
    try:
        embedding = await embed.embed_image(data)
    except EmbedUnavailable as exc:  # H7
        raise HTTPException(
            status_code=503, detail="Tìm ảnh đang khởi động, thử lại"
        ) from exc

    product_ids = await search_svc.search(embedding)
    return {"product_ids": product_ids}  # H1
```

- [ ] **Step 3c: Wire into `app/main.py`**

Add after line 16 (`from app.routers import products as products_router`):

```python
from app.routers import search as search_router
```

Add after line 21 (`from app.services.product_client import ProductClient`):

```python
from app.services.embed_client import EmbedClient
from app.services.image_search import ImageSearchService
```

Add after line 38 (`product_client = ProductClient(http_client)`):

```python
    embed_client = EmbedClient(http_client)
```

Add after line 55 (the `app.state.description = ...` assignment block, before `await consumer.start()`):

```python
    app.state.embed = embed_client
    app.state.image_search = ImageSearchService(pool)
```

Add after line 83 (`app.include_router(products_router.router)`):

```python
app.include_router(search_router.router)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_search_router.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Format, lint, commit**

```bash
uv run ruff format . && uv run ruff check .
git add app/routers/search.py app/main.py pyproject.toml uv.lock tests/test_search_router.py
git commit -m "feat(search): add POST /search/image (multipart file, per-IP rate limit, 503 on embed down)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: ImageIndexer + ProductClient.get_product_images + main.py wiring

**Files:**
- Modify: `app/services/product_client.py:33-40` (add `get_product_images` after `check_inventory`)
- Create: `app/services/image_indexer.py`
- Modify: `app/main.py:39` (create `image_indexer` right after `indexer = ProductIndexer(...)`), `:21` area (add `ImageIndexer` import), `:49` area (add `app.state.image_indexer`)
- Test: `tests/test_image_indexer.py` (create)

Atomic DELETE+INSERT pattern mirrors `app/services/indexer.py:75-93`; the X-Internal-Key GET mirrors `app/services/product_client.py:15-22`.

**Interfaces:**
- Consumes: `EmbedClient.embed_image(bytes) -> list[float]` + `EmbedUnavailable` (Task 2); `settings.image_max_upload_bytes` (Task 1); shop-api `GET /api/internal/products/{id}/images -> [{"resourceId": int, "url": str, "sortOrder": int}]`
- Produces: `ProductClient.get_product_images(self, product_id: int) -> list[dict]`; `class ImageIndexer: def __init__(self, pool, embed_client, product_client, http_client); async def index_product_images(self, product_id: int) -> int` (count of embedded images; 0 = nothing replaced, existing rows kept — H9); `async def delete_product_images(self, product_id: int) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_image_indexer.py
from unittest.mock import AsyncMock

import httpx
import pytest

from app.services.embed_client import EmbedUnavailable
from app.services.image_indexer import ImageIndexer


def _http_client() -> httpx.AsyncClient:
    # Every image URL "downloads" to the same small byte payload.
    return httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(200, content=b"imgbytes")
        )
    )


@pytest.mark.asyncio
async def test_one_image_fails_others_still_indexed(
    truncate_image_embeddings, make_embedding_768
):
    pool = truncate_image_embeddings
    pc = AsyncMock()
    pc.get_product_images.return_value = [
        {"resourceId": 10, "url": "http://cdn/a.jpg", "sortOrder": 0},
        {"resourceId": 11, "url": "http://cdn/b.jpg", "sortOrder": 1},
    ]
    embed = AsyncMock()
    # First image fails to embed, second succeeds.
    embed.embed_image.side_effect = [
        EmbedUnavailable("bad"),
        make_embedding_768({0: 1.0}),
    ]

    async with _http_client() as http:
        indexer = ImageIndexer(pool, embed, pc, http)
        n = await indexer.index_product_images(5)

    assert n == 1  # only the second image embedded (H9 resilient)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT resource_id FROM product_image_embeddings WHERE product_id = 5"
        )
    assert [r["resource_id"] for r in rows] == [11]


@pytest.mark.asyncio
async def test_all_images_fail_does_not_wipe_existing(
    truncate_image_embeddings, make_embedding_768
):
    from tests.conftest import seed_image_embedding

    pool = truncate_image_embeddings
    # Pre-existing indexed row for product 5.
    await seed_image_embedding(pool, 99, 5, "http://cdn/old.jpg", make_embedding_768({0: 1.0}))

    pc = AsyncMock()
    pc.get_product_images.return_value = [
        {"resourceId": 10, "url": "http://cdn/a.jpg", "sortOrder": 0},
    ]
    embed = AsyncMock()
    embed.embed_image.side_effect = EmbedUnavailable("bad")  # ALL fail

    async with _http_client() as http:
        indexer = ImageIndexer(pool, embed, pc, http)
        n = await indexer.index_product_images(5)

    assert n == 0
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT resource_id FROM product_image_embeddings WHERE product_id = 5"
        )
    assert [r["resource_id"] for r in rows] == [99]  # existing row NOT wiped (H9)


@pytest.mark.asyncio
async def test_delete_product_images_removes_rows(
    truncate_image_embeddings, make_embedding_768
):
    from tests.conftest import seed_image_embedding

    pool = truncate_image_embeddings
    await seed_image_embedding(pool, 99, 5, "http://cdn/x.jpg", make_embedding_768({0: 1.0}))

    async with _http_client() as http:
        indexer = ImageIndexer(pool, AsyncMock(), AsyncMock(), http)
        deleted = await indexer.delete_product_images(5)

    assert deleted == 1
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT resource_id FROM product_image_embeddings WHERE product_id = 5"
        )
    assert rows == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `DATABASE_URL=postgresql://postgres:postgres@localhost:5432/goodminton_test uv run pytest tests/test_image_indexer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.image_indexer'`

- [ ] **Step 3a: Add `get_product_images` to `app/services/product_client.py`**

Insert after the `check_inventory` method (after line 40):

```python

    async def get_product_images(self, product_id: int) -> list[dict]:
        r = await self._client.get(
            f"{settings.shop_api_url}/api/internal/products/{product_id}/images",
            headers=self._headers(),
            timeout=10.0,
        )
        r.raise_for_status()
        return r.json()
```

- [ ] **Step 3b: Write `app/services/image_indexer.py`**

```python
# app/services/image_indexer.py
"""Image indexer — downloads a product's images, embeds them, atomically replaces rows.

Read-only against the shop-api catalog: only reads image URLs; writes ONLY
product_image_embeddings.
"""

import logging

import asyncpg
import httpx

from app.core.config import settings
from app.services.embed_client import EmbedClient
from app.services.product_client import ProductClient

log = logging.getLogger(__name__)


class ImageIndexer:
    def __init__(
        self,
        pool: asyncpg.Pool,
        embed_client: EmbedClient,
        product_client: ProductClient,
        http_client: httpx.AsyncClient,
    ):
        self._pool = pool
        self._embed = embed_client
        self._client = product_client
        self._http = http_client

    async def _download(self, url: str) -> bytes:
        r = await self._http.get(url, timeout=30.0)
        r.raise_for_status()
        data = r.content
        if len(data) > settings.image_max_upload_bytes:  # cap decoded size (H9)
            raise ValueError(f"image exceeds byte cap: {url}")
        return data

    async def index_product_images(self, product_id: int) -> int:
        """Embed every image; atomic replace over successes. Returns embedded count.

        Per-image failures are skipped (H9). If ALL fail, existing rows are kept
        (no wipe) and 0 is returned.
        """
        images = await self._client.get_product_images(product_id)
        embedded: list[tuple[int, str, list[float]]] = []
        for img in images:
            resource_id = img["resourceId"]
            url = img["url"]
            try:
                data = await self._download(url)
                embedding = await self._embed.embed_image(data)
            except Exception:
                log.exception(
                    "Failed to embed image %s for product %s", url, product_id
                )
                continue
            embedded.append((resource_id, url, embedding))

        if not embedded:  # H9: never wipe existing rows on total failure
            log.warning(
                "No images embedded for product %s — keeping existing rows", product_id
            )
            return 0

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "DELETE FROM product_image_embeddings WHERE product_id = $1",
                    product_id,
                )
                for resource_id, url, embedding in embedded:
                    await conn.execute(
                        "INSERT INTO product_image_embeddings "
                        "(resource_id, product_id, url, embedding) "
                        "VALUES ($1, $2, $3, $4)",
                        resource_id,
                        product_id,
                        url,
                        embedding,
                    )

        log.info("Indexed %d images for product %s", len(embedded), product_id)
        return len(embedded)

    async def delete_product_images(self, product_id: int) -> int:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM product_image_embeddings WHERE product_id = $1",
                product_id,
            )
        log.info("Deleted image embeddings for product %s (%s)", product_id, result)
        try:
            return int(result.split()[-1])  # asyncpg returns "DELETE N"
        except (ValueError, IndexError):
            return 0
```

- [ ] **Step 3c: Wire into `app/main.py`**

Add to the imports (after line 21's `from app.services.image_search import ImageSearchService`, added in Task 4):

```python
from app.services.image_indexer import ImageIndexer
```

Add immediately after line 39 (`indexer = ProductIndexer(pool, embedding, product_client)`) — it must exist before the `ProductConsumer(...)` line so Task 6 can pass it:

```python
    image_indexer = ImageIndexer(pool, embed_client, product_client, http_client)
```

Add to the `app.state` block (near line 49, `app.state.indexer = indexer`):

```python
    app.state.image_indexer = image_indexer
```

- [ ] **Step 4: Run test to verify it passes**

Run: `DATABASE_URL=postgresql://postgres:postgres@localhost:5432/goodminton_test uv run pytest tests/test_image_indexer.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Format, lint, commit**

```bash
uv run ruff format . && uv run ruff check .
git add app/services/image_indexer.py app/services/product_client.py app/main.py tests/test_image_indexer.py
git commit -m "feat(index): add ImageIndexer (resilient per-image, no-wipe-on-all-fail) + get_product_images

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Consumer hook — re-index images on `fieldsChanged` = images

**Files:**
- Modify: `app/messaging/product_consumer.py:17-20` (add `image_indexer` to `__init__`), `:53-77` (`_handle`: image re-index on `"images"` field; image delete on `"deleted"`)
- Modify: `app/main.py:40` (`ProductConsumer(indexer)` -> `ProductConsumer(indexer, image_indexer)`)
- Test: `tests/test_product_consumer_images.py` (create)

**Interfaces:**
- Consumes: `ImageIndexer.index_product_images(int) -> int`, `ImageIndexer.delete_product_images(int) -> int` (Task 5)
- Produces: `ProductConsumer.__init__(self, indexer, image_indexer)` (new required 2nd arg); `_handle` triggers `image_indexer.index_product_images(product_id)` when `"images"` ∈ `fieldsChanged`, independently of the existing `SEMANTIC_FIELDS` text-reindex.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_product_consumer_images.py
from unittest.mock import AsyncMock

import pytest

from app.messaging.product_consumer import ProductConsumer


@pytest.mark.asyncio
async def test_images_field_triggers_image_reindex_only():
    idx = AsyncMock()
    img = AsyncMock()
    consumer = ProductConsumer(idx, img)

    await consumer._handle(
        {"action": "updated", "productId": 5, "fieldsChanged": ["images"]}
    )

    img.index_product_images.assert_awaited_once_with(5)
    idx.index_product.assert_not_called()  # 'images' is not a SEMANTIC_FIELD


@pytest.mark.asyncio
async def test_semantic_and_images_triggers_both():
    idx = AsyncMock()
    img = AsyncMock()
    consumer = ProductConsumer(idx, img)

    await consumer._handle(
        {"action": "updated", "productId": 5, "fieldsChanged": ["name", "images"]}
    )

    idx.index_product.assert_awaited_once_with(5)
    img.index_product_images.assert_awaited_once_with(5)


@pytest.mark.asyncio
async def test_deleted_removes_text_and_image_rows():
    idx = AsyncMock()
    img = AsyncMock()
    consumer = ProductConsumer(idx, img)

    await consumer._handle({"action": "deleted", "productId": 5})

    idx.delete_product.assert_awaited_once_with(5)
    img.delete_product_images.assert_awaited_once_with(5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_product_consumer_images.py -v`
Expected: FAIL with `TypeError: __init__() missing 1 required positional argument: 'image_indexer'`

- [ ] **Step 3a: Update `ProductConsumer.__init__` (`app/messaging/product_consumer.py`)**

Replace lines 18-20:

```python
    def __init__(self, indexer: ProductIndexer):
        self._indexer = indexer
        self._connection: aio_pika.abc.AbstractRobustConnection | None = None
```

with:

```python
    def __init__(self, indexer: ProductIndexer, image_indexer: ImageIndexer):
        self._indexer = indexer
        self._image_indexer = image_indexer
        self._connection: aio_pika.abc.AbstractRobustConnection | None = None
```

Add the import after line 10 (`from app.services.indexer import ProductIndexer`):

```python
from app.services.image_indexer import ImageIndexer
```

- [ ] **Step 3b: Update `_handle` (replace lines 62-77)**

Replace:

```python
        if action == "deleted":
            await self._indexer.delete_product(product_id)
            return

        if action in ("created", "updated"):
            if not fields & SEMANTIC_FIELDS:
                log.debug(
                    "Skip product %s — no semantic fields changed (%s)",
                    product_id,
                    fields,
                )
                return
            await self._indexer.index_product(product_id)
            return

        log.warning("Unknown action '%s' for product %s", action, product_id)
```

with:

```python
        if action == "deleted":
            await self._indexer.delete_product(product_id)
            await self._image_indexer.delete_product_images(product_id)
            return

        if action in ("created", "updated"):
            if fields & SEMANTIC_FIELDS:
                await self._indexer.index_product(product_id)
            if "images" in fields:
                await self._image_indexer.index_product_images(product_id)
            if not (fields & SEMANTIC_FIELDS) and "images" not in fields:
                log.debug(
                    "Skip product %s — no indexed fields changed (%s)",
                    product_id,
                    fields,
                )
            return

        log.warning("Unknown action '%s' for product %s", action, product_id)
```

- [ ] **Step 3c: Update `app/main.py`**

Change line 40 from:

```python
    consumer = ProductConsumer(indexer)
```

to:

```python
    consumer = ProductConsumer(indexer, image_indexer)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_product_consumer_images.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Format, lint, commit**

```bash
uv run ruff format . && uv run ruff check .
git add app/messaging/product_consumer.py app/main.py tests/test_product_consumer_images.py
git commit -m "feat(consumer): re-index images when fieldsChanged includes 'images'

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Backfill script `scripts/backfill_product_images.py`

**Files:**
- Create: `scripts/backfill_product_images.py`
- Test: `tests/test_backfill_images.py` (create)

Mirrors `scripts/backfill_products.py` structure and **reuses** its `fetch_product_ids` (the `SELECT id FROM products WHERE is_visible = true ORDER BY id` query, `scripts/backfill_products.py:45-50`) and `resolve_database_url`.

**Interfaces:**
- Consumes: `scripts.backfill_products.fetch_product_ids(conn) -> list[int]`, `scripts.backfill_products.resolve_database_url() -> str`; `ImageIndexer` (Task 5); `EmbedClient` (Task 2)
- Produces: `scripts/backfill_product_images.py` with `fetch_product_ids` re-exported (the same object) + an `async def main()` that indexes each visible product's images.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backfill_images.py
def test_reuses_visible_product_id_query():
    # The image backfill must REUSE the products.is_visible id source, not roll its own.
    from scripts import backfill_product_images, backfill_products

    assert (
        backfill_product_images.fetch_product_ids
        is backfill_products.fetch_product_ids
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_backfill_images.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.backfill_product_images'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/backfill_product_images.py
"""One-time backfill: embed all visible products' images into product_image_embeddings.

Reuses the visible-product id source from backfill_products (SELECT id FROM products
WHERE is_visible = true). Idempotent — re-running just re-embeds.

Run:
    docker compose -f docker-compose.prod.yml exec -T rag-service \\
        uv run python scripts/backfill_product_images.py
"""

import asyncio
import logging
import os
import sys

import asyncpg
import httpx
from pgvector.asyncpg import register_vector

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.embed_client import EmbedClient  # noqa: E402
from app.services.image_indexer import ImageIndexer  # noqa: E402
from app.services.product_client import ProductClient  # noqa: E402
from scripts.backfill_products import (  # noqa: E402
    fetch_product_ids,
    resolve_database_url,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


async def main() -> None:
    pool = await asyncpg.create_pool(
        dsn=resolve_database_url(),
        init=lambda conn: register_vector(conn),
    )

    async with httpx.AsyncClient() as http_client:
        embed = EmbedClient(http_client)
        product_client = ProductClient(http_client)
        indexer = ImageIndexer(pool, embed, product_client, http_client)

        async with pool.acquire() as conn:
            ids = await fetch_product_ids(conn)
        log.info("Found %d visible products to index images for", len(ids))

        succeeded = 0
        failed = 0
        for pid in ids:
            try:
                await indexer.index_product_images(pid)
                succeeded += 1
            except Exception:
                log.exception("Failed to index images for product %s", pid)
                failed += 1

        log.info("Done. Succeeded: %d | Failed: %d", succeeded, failed)

    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_backfill_images.py -v`
Expected: PASS

- [ ] **Step 5: Format, lint, commit**

```bash
uv run ruff format . && uv run ruff check .
git add scripts/backfill_product_images.py tests/test_backfill_images.py
git commit -m "feat(backfill): add backfill_product_images reusing is_visible id source

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Final Verification

- [ ] Run the whole suite: `DATABASE_URL=postgresql://postgres:postgres@localhost:5432/goodminton_test uv run pytest`  Expected: all pass (DB-backed image tests run; none skipped when `DATABASE_URL` is set).
- [ ] `uv run ruff format --check .`  Expected: no reformat needed.
- [ ] `uv run ruff check .`  Expected: no lint errors.
