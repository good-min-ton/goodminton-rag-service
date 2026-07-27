# Chatbot Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the sales chatbot category-aware and context-aware, and bind product cards to structured backend output — closing audit problems P1 (wrong-category recommendations), P2 (no conversation context), P5 (text/cards out of sync).

**Architecture:** Insert a deterministic query-understanding stage before retrieval (hybrid intent/category extraction + contextualization from server-side session state in Redis), add a category filter + multi-category quota to the pgvector search, and replace prose-scraped product cards with a `display_products` field the backend computes from structured retrieval/tool output. The LLM + tool loop is unchanged; category selection is never delegated to the model.

**Tech Stack:** FastAPI, asyncpg + pgvector, httpx→Ollama, `redis.asyncio` (NEW dependency), pydantic-settings; Next.js 16 + TypeScript (shop-ui).

**Spec:** `docs/superpowers/specs/2026-07-27-chatbot-foundation-design.md`
**Audit:** `docs/superpowers/audits/2026-07-27-chatbot-sales-flow-audit.md`

## Global Constraints

- Branch off `origin/main` in each repo: RAG `feat/chatbot-foundation` (already created), shop-ui `feat/chatbot-foundation-ui` (create in Task 6, off `origin/main` = `dae04d2`).
- RAG stack is hand-written — NO LangChain except the existing `langchain-text-splitters`. LLM/embeddings via httpx→Ollama; vector search via raw pgvector SQL (`<=>`).
- `kb_chunks` already has a `metadata JSONB DEFAULT '{}'` column and embeds `VECTOR(1024)` (bge-m3). Category value = the raw `product['category']` string used at index time (`indexer.py:44-46`).
- **Redis is a NEW dependency for RAG** — must degrade gracefully: if Redis is unavailable, the endpoint behaves as today (stateless), never 500.
- Never add a cosine distance threshold (keep the "no threshold" philosophy — H3 in the audit).
- Response-contract additions are ADDITIVE: keep `answer`, `sources`, `products`, `order_draft`; add `intent`, `categories`, `display_products`, `conversation_state`. `display_products` supersedes `products` for card rendering.
- Session lifetime: every page reload = a fresh conversation (new `session_id`, empty history).
- Non-goals (do NOT build here): order state machine (Spec 2 — only reserve `selected_product_id`/`order_status` fields), tool-result JSON sanitize (Spec 3), image-search path (`search.py`, `image_*`). Keep `prepare_order` read-only.
- RAG tests: `pytest`; DB-backed tests use the `truncate_kb`/`pg_pool` fixtures in `tests/conftest.py` and SKIP when `DATABASE_URL` is unset. Lint: `ruff check` + `ruff format --check`.
- shop-ui: no test runner — gate is `npx tsc --noEmit` (exit 0).

---

### Task 1: Conversation state model + Redis-backed store (with graceful degrade)

**Files:**
- Create: `app/services/conversation_state.py`
- Modify: `app/core/config.py` (add settings after line 26, the Retrieval block)
- Modify: `app/main.py` (wire client + store into `app.state`, lifespan)
- Test: `tests/test_conversation_state.py`

**Interfaces:**
- Produces:
  - `class ConversationState(BaseModel)` with fields `intent: str | None = None`, `categories: list[str] = []`, `price_preference: str | None = None`, `selected_product_id: int | None = None`, `order_status: str | None = None`.
  - `class ConversationStateStore` with `__init__(self, client)` where `client` is a `redis.asyncio.Redis | None`; `async def load(self, session_id: str | None) -> ConversationState`; `async def save(self, session_id: str | None, state: ConversationState) -> None`.
  - `settings.redis_url: str | None`, `settings.chat_state_ttl_seconds: int = 3600`, `settings.chat_display_products_max: int = 4`.

- [ ] **Step 1: Add config knobs**

In `app/core/config.py`, after the Retrieval block (`min_query_length`, line 26), add:

```python
    # Chatbot conversation state (Redis) + display cards
    redis_url: str | None = None  # e.g. redis://:pass@redis:6379/0 ; None => stateless degrade
    chat_state_ttl_seconds: int = 3600
    chat_display_products_max: int = 4
```

- [ ] **Step 2: Add `redis` dependency**

Run: `uv add redis` (adds `redis>=5` with asyncio support to `pyproject.toml`). Verify it imports: `uv run python -c "import redis.asyncio as r; print(r.Redis)"`.

- [ ] **Step 3: Write the failing tests**

`tests/test_conversation_state.py`:

```python
import pytest
from app.services.conversation_state import ConversationState, ConversationStateStore


class _FakeRedis:
    """Minimal in-memory stand-in for redis.asyncio.Redis (get/set/expire)."""

    def __init__(self, raise_on=None):
        self._data: dict[str, str] = {}
        self._raise_on = raise_on  # set to "get"/"set" to simulate an outage

    async def get(self, key):
        if self._raise_on == "get":
            raise ConnectionError("redis down")
        return self._data.get(key)

    async def set(self, key, value, ex=None):
        if self._raise_on == "set":
            raise ConnectionError("redis down")
        self._data[key] = value


async def test_save_then_load_roundtrips_state():
    store = ConversationStateStore(_FakeRedis())
    state = ConversationState(intent="buy", categories=["pants"], price_preference="cheapest")
    await store.save("sess-1", state)
    loaded = await store.load("sess-1")
    assert loaded.categories == ["pants"]
    assert loaded.price_preference == "cheapest"
    assert loaded.intent == "buy"


async def test_load_missing_session_returns_fresh_state():
    store = ConversationStateStore(_FakeRedis())
    loaded = await store.load("never-seen")
    assert loaded == ConversationState()


async def test_none_session_id_returns_fresh_and_save_is_noop():
    store = ConversationStateStore(_FakeRedis())
    await store.save(None, ConversationState(categories=["shoes"]))
    assert await store.load(None) == ConversationState()


async def test_no_client_degrades_to_stateless():
    store = ConversationStateStore(None)
    await store.save("sess-1", ConversationState(categories=["pants"]))  # no-op, no raise
    assert await store.load("sess-1") == ConversationState()


async def test_redis_outage_degrades_without_raising():
    store = ConversationStateStore(_FakeRedis(raise_on="get"))
    assert await store.load("sess-1") == ConversationState()  # swallows ConnectionError
    store2 = ConversationStateStore(_FakeRedis(raise_on="set"))
    await store2.save("sess-1", ConversationState(categories=["pants"]))  # no raise
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `uv run pytest tests/test_conversation_state.py -v`
Expected: FAIL with `ModuleNotFoundError: app.services.conversation_state`.

- [ ] **Step 5: Implement the store**

`app/services/conversation_state.py`:

```python
"""Server-side chat conversation state, keyed by session_id, stored in Redis.

Degrades to stateless (fresh state, no-op save) when Redis is absent or down —
the chat endpoint must never 500 because of state I/O.
"""

import logging

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

_KEY_PREFIX = "chat:state:"


class ConversationState(BaseModel):
    intent: str | None = None
    categories: list[str] = Field(default_factory=list)
    price_preference: str | None = None
    # RESERVED for Spec 2 (order flow) — unused in the Foundation.
    selected_product_id: int | None = None
    order_status: str | None = None


class ConversationStateStore:
    def __init__(self, client) -> None:
        # client: redis.asyncio.Redis | None
        self._client = client
        self._ttl = None  # set by main.py from settings.chat_state_ttl_seconds

    def with_ttl(self, ttl_seconds: int) -> "ConversationStateStore":
        self._ttl = ttl_seconds
        return self

    async def load(self, session_id: str | None) -> ConversationState:
        if not session_id or self._client is None:
            return ConversationState()
        try:
            raw = await self._client.get(_KEY_PREFIX + session_id)
        except Exception as exc:  # redis down / decode error — degrade
            log.warning("state load failed (%s); using fresh state", exc)
            return ConversationState()
        if not raw:
            return ConversationState()
        try:
            return ConversationState.model_validate_json(raw)
        except Exception as exc:
            log.warning("state parse failed (%s); using fresh state", exc)
            return ConversationState()

    async def save(self, session_id: str | None, state: ConversationState) -> None:
        if not session_id or self._client is None:
            return
        try:
            await self._client.set(
                _KEY_PREFIX + session_id,
                state.model_dump_json(),
                ex=self._ttl,
            )
        except Exception as exc:  # redis down — degrade silently
            log.warning("state save failed (%s); continuing stateless", exc)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_conversation_state.py -v`
Expected: PASS (5 tests).

- [ ] **Step 7: Wire the store into `app.state`**

In `app/main.py`, inside `lifespan` (after `pool = await create_pool()`, line 34) add:

```python
    import redis.asyncio as aioredis  # local import keeps module import cheap

    redis_client = (
        aioredis.from_url(settings.redis_url, decode_responses=True)
        if settings.redis_url
        else None
    )
```

After the other `app.state.*` assignments (near line 55) add:

```python
    from app.services.conversation_state import ConversationStateStore

    app.state.redis = redis_client
    app.state.conversation_state = ConversationStateStore(redis_client).with_ttl(
        settings.chat_state_ttl_seconds
    )
```

In the shutdown section (after `yield`, near line 61) add before `await pool.close()`:

```python
    if redis_client is not None:
        await redis_client.aclose()
```

- [ ] **Step 8: Verify wiring imports cleanly + lint**

Run: `uv run python -c "import app.main"` (Expected: no error).
Run: `uv run ruff check app/services/conversation_state.py app/main.py app/core/config.py && uv run ruff format --check app/services/conversation_state.py`
Expected: clean.

- [ ] **Step 9: Commit**

```bash
git add app/services/conversation_state.py app/core/config.py app/main.py tests/test_conversation_state.py pyproject.toml uv.lock
git commit -m "feat(chat): Redis-backed conversation state store with graceful degrade

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Query understanding (hybrid intent/category/price extraction + contextualization)

**Files:**
- Create: `app/services/query_understanding.py`
- Modify: `app/main.py` (wire into `app.state`)
- Test: `tests/test_query_understanding.py`

**Interfaces:**
- Consumes: `ConversationState` (Task 1); `LLMService` (existing `app/services/llm.py`, method `async def chat(self, messages: list[dict]) -> str`).
- Produces:
  - `class QueryUnderstanding(BaseModel)` with `intent: str | None`, `categories: list[str]`, `price_preference: str | None`, `retrieval_query: str`.
  - `class QueryUnderstandingService.__init__(self, llm)`; `async def analyze(self, message: str, state: ConversationState) -> QueryUnderstanding`.
  - Module constant `CATEGORY_KEYWORDS: dict[str, list[str]]` mapping a stored category string → trigger keywords.

- [ ] **Step 1: Discover the real category values (grounding step, not code)**

The category filter later matches `metadata->>'category'` against these exact strings, so the map's KEYS must equal the real `product['category']` values. Determine them from shop-api or the DB, e.g.:
`SELECT DISTINCT metadata->>'category' FROM kb_chunks WHERE doc_type='product';` (after Task 4 backfill) or the shop-api category list.
Use the discovered strings as the dict keys below. The values shown (`"Vợt cầu lông"`, `"Giày cầu lông"`, `"Quần cầu lông"`, `"Áo cầu lông"`, `"Phụ kiện"`) are the expected canonical names — **replace with the exact live values if they differ.**

- [ ] **Step 2: Write the failing tests**

`tests/test_query_understanding.py`:

```python
from unittest.mock import AsyncMock

from app.services.conversation_state import ConversationState
from app.services.query_understanding import QueryUnderstandingService


def _svc(llm=None):
    return QueryUnderstandingService(llm or AsyncMock())


async def test_single_category_pants():
    qu = await _svc().analyze("mua quần cầu lông", ConversationState())
    assert qu.categories == ["Quần cầu lông"]
    assert qu.retrieval_query == "mua quần cầu lông"


async def test_multi_category_pants_and_shoes():
    qu = await _svc().analyze("cho tôi xem quần và giày", ConversationState())
    assert set(qu.categories) == {"Quần cầu lông", "Giày cầu lông"}


async def test_elliptical_inherits_categories_from_state():
    state = ConversationState(categories=["Quần cầu lông"])
    qu = await _svc().analyze("rẻ nhất", state)
    assert qu.categories == ["Quần cầu lông"]
    assert qu.price_preference == "cheapest"
    # contextualized query carries the inherited scope so the vector search is scoped
    assert "Quần cầu lông" in qu.retrieval_query


async def test_price_preference_cheapest_keywords():
    qu = await _svc().analyze("quần nào rẻ nhất", ConversationState())
    assert qu.price_preference == "cheapest"


async def test_no_rule_match_falls_back_to_llm():
    llm = AsyncMock()
    llm.chat.return_value = "Giày cầu lông"  # LLM returns a category label
    qu = await _svc(llm).analyze("đôi nào bền cho người mới", ConversationState())
    assert qu.categories == ["Giày cầu lông"]
    llm.chat.assert_awaited_once()


async def test_llm_fallback_unusable_output_yields_no_category():
    llm = AsyncMock()
    llm.chat.return_value = "tôi không rõ"
    qu = await _svc(llm).analyze("asdfqwer", ConversationState())
    assert qu.categories == []  # unfiltered retrieval downstream (current behavior)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_query_understanding.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Implement the service**

`app/services/query_understanding.py`:

```python
"""Deterministic query understanding: extract intent/category/price from the
message (rule-based first, LLM fallback), and contextualize elliptical follow-ups
using prior conversation state. Category selection is NEVER left to the tool loop.
"""

import logging

from pydantic import BaseModel, Field

from app.services.conversation_state import ConversationState

log = logging.getLogger(__name__)

# KEY = the exact `product['category']` string stored in kb_chunks.metadata.
# VALUE = trigger keywords (lowercase, unaccented-tolerant substrings).
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Vợt cầu lông": ["vợt", "vot", "racket", "racquet"],
    "Giày cầu lông": ["giày", "giay", "shoe", "sneaker", "đôi giày", "doi giay"],
    "Quần cầu lông": ["quần", "quan", "short", "shorts"],
    "Áo cầu lông": ["áo", "ao", "shirt", "jersey"],
    "Phụ kiện": ["phụ kiện", "phu kien", "cước", "cuoc", "grip", "quấn cán", "túi", "balo"],
}

_PRICE_CHEAPEST = ["rẻ nhất", "re nhat", "rẻ", "re ", "giá thấp", "gia thap", "rẻ hơn", "re hon"]
_INTENT_PRICE = ["giá", "gia ", "bao nhiêu", "bao nhieu", "giảm giá", "sale"]
_INTENT_STOCK = ["còn hàng", "con hang", "hết hàng", "het hang", "size", "tồn kho", "ton kho"]
_INTENT_BUY = ["mua", "đặt", "dat ", "order", "chốt", "lấy"]


class QueryUnderstanding(BaseModel):
    intent: str | None = None
    categories: list[str] = Field(default_factory=list)
    price_preference: str | None = None
    retrieval_query: str = ""


class QueryUnderstandingService:
    def __init__(self, llm) -> None:
        self._llm = llm  # LLMService

    def _rule_categories(self, low: str) -> list[str]:
        out: list[str] = []
        for category, kws in CATEGORY_KEYWORDS.items():
            if any(kw in low for kw in kws):
                out.append(category)
        return out

    def _rule_intent(self, low: str) -> str | None:
        if any(k in low for k in _INTENT_STOCK):
            return "stock"
        if any(k in low for k in _INTENT_PRICE):
            return "price"
        if any(k in low for k in _INTENT_BUY):
            return "buy"
        return "browse"

    async def _llm_category(self, message: str) -> list[str]:
        prompt = (
            "Phân loại câu sau vào MỘT danh mục sản phẩm cầu lông, trả lời DUY NHẤT "
            "một trong: Vợt cầu lông | Giày cầu lông | Quần cầu lông | Áo cầu lông | "
            "Phụ kiện | KHÔNG RÕ. Câu: " + message
        )
        try:
            raw = (await self._llm.chat([{"role": "user", "content": prompt}])).strip()
        except Exception as exc:
            log.warning("LLM category fallback failed (%s)", exc)
            return []
        for category in CATEGORY_KEYWORDS:
            if category.lower() in raw.lower():
                return [category]
        return []

    async def analyze(self, message: str, state: ConversationState) -> QueryUnderstanding:
        low = message.lower()
        categories = self._rule_categories(low)
        intent = self._rule_intent(low)
        price_pref = "cheapest" if any(k in low for k in _PRICE_CHEAPEST) else None

        if not categories:
            # LLM fallback for category only; if it also fails, inherit from state.
            categories = await self._llm_category(message)
        if not categories and state.categories:
            categories = list(state.categories)  # elliptical follow-up inherits scope

        # Contextualize the retrieval query: append inherited scope so the vector
        # search is anchored even when the message itself is elliptical ("rẻ nhất").
        retrieval_query = message.strip()
        if categories and not self._rule_categories(low):
            retrieval_query = f"{message.strip()} {' '.join(categories)}"

        return QueryUnderstanding(
            intent=intent,
            categories=categories,
            price_preference=price_pref,
            retrieval_query=retrieval_query,
        )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_query_understanding.py -v`
Expected: PASS (6 tests). If the discovered category strings (Step 1) differ from the defaults, update both `CATEGORY_KEYWORDS` keys and the test expectations together.

- [ ] **Step 6: Wire into `app.state`**

In `app/main.py` lifespan, after `app.state.llm = LLMService(http_client)` (line 48) add:

```python
    from app.services.query_understanding import QueryUnderstandingService

    app.state.query_understanding = QueryUnderstandingService(app.state.llm)
```

- [ ] **Step 7: Lint + import check**

Run: `uv run ruff check app/services/query_understanding.py && uv run ruff format --check app/services/query_understanding.py && uv run python -c "import app.main"`
Expected: clean, no error.

- [ ] **Step 8: Commit**

```bash
git add app/services/query_understanding.py app/main.py tests/test_query_understanding.py
git commit -m "feat(chat): hybrid query understanding (intent/category/price + contextualize)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Category-filtered retrieval with multi-category quota

**Files:**
- Modify: `app/services/retrieval.py` (the `search` method, lines 23-51)
- Test: `tests/test_retrieval_category.py`

**Interfaces:**
- Consumes: `Chunk` (existing), `settings.retrieval_top_k`.
- Produces: `RetrievalService.search(self, query_embedding, k=None, categories: list[str] | None = None, doc_type: str | None = None) -> list[Chunk]`. When `categories` has ≥2 entries, results are a per-category quota merge so every requested category is represented.

- [ ] **Step 1: Write the failing DB tests**

`tests/test_retrieval_category.py` (uses conftest fixtures; skips without `DATABASE_URL`):

```python
import pytest
from app.services.retrieval import RetrievalService


async def _seed(pool, make_embedding):
    from tests.conftest import seed_chunk
    # 3 products in category A, 1 in B; distinct embedding dims for deterministic order.
    rows = [
        ("101", "Quần cầu lông", {0: 1.0}),
        ("102", "Quần cầu lông", {1: 1.0}),
        ("103", "Quần cầu lông", {2: 1.0}),
        ("201", "Giày cầu lông", {3: 1.0}),
    ]
    async with pool.acquire() as conn:
        for sid, cat, dims in rows:
            await conn.execute(
                "INSERT INTO kb_chunks (doc_type, source_id, chunk_index, content, metadata, embedding) "
                "VALUES ('product', $1, 0, $2, $3::jsonb, $4)",
                sid, f"Sản phẩm: P{sid}", f'{{"category": "{cat}"}}', make_embedding(dims),
            )


async def test_category_filter_restricts_results(truncate_kb, make_embedding):
    pool = truncate_kb
    await _seed(pool, make_embedding)
    svc = RetrievalService(pool)
    q = make_embedding({3: 1.0})  # closest to the shoe row
    chunks = await svc.search(q, categories=["Quần cầu lông"])
    assert chunks, "expected pants results"
    assert all(c.source_id in {"101", "102", "103"} for c in chunks)  # no shoe leaked in


async def test_multi_category_returns_both(truncate_kb, make_embedding):
    pool = truncate_kb
    await _seed(pool, make_embedding)
    svc = RetrievalService(pool)
    q = make_embedding({0: 1.0})
    chunks = await svc.search(q, k=4, categories=["Quần cầu lông", "Giày cầu lông"])
    cats_present = {c.source_id for c in chunks}
    assert "201" in cats_present  # the single shoe is present despite 3 closer pants
    assert cats_present & {"101", "102", "103"}  # pants present too


async def test_no_category_preserves_global_search(truncate_kb, make_embedding):
    pool = truncate_kb
    await _seed(pool, make_embedding)
    svc = RetrievalService(pool)
    chunks = await svc.search(make_embedding({0: 1.0}), k=4)
    assert len(chunks) == 4  # unfiltered, current behavior
```

- [ ] **Step 2: Run to verify they fail**

Run: `DATABASE_URL=postgresql://admin:postgresql123@localhost:5433/goodminton_test uv run pytest tests/test_retrieval_category.py -v`
Expected: FAIL (`search()` got an unexpected keyword argument `categories`).

- [ ] **Step 3: Implement the filter + quota**

Replace `RetrievalService.search` in `app/services/retrieval.py` with:

```python
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
                merged.extend(await self._search_one(query_embedding, per, [cat], doc_type))
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
```

- [ ] **Step 4: Run to verify they pass**

Run: `DATABASE_URL=postgresql://admin:postgresql123@localhost:5433/goodminton_test uv run pytest tests/test_retrieval_category.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Regression + lint**

Run: `DATABASE_URL=... uv run pytest tests/ -q` (existing retrieval/similar tests still green) and `uv run ruff check app/services/retrieval.py`.
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add app/services/retrieval.py tests/test_retrieval_category.py
git commit -m "feat(retrieval): category filter + multi-category quota merge

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Index category into metadata + backfill existing chunks

**Files:**
- Modify: `app/services/indexer.py` (`index_product`, lines 68-96)
- Create: `scripts/backfill_chunk_metadata.py`
- Test: `tests/test_indexer_metadata.py`

**Interfaces:**
- Consumes: existing `ProductIndexer`, `product['category']`.
- Produces: every product chunk row written with `metadata = {"category": <product['category']>}`. Backfill re-indexes all product `source_id`s so already-stored rows gain the metadata.

- [ ] **Step 1: Write the failing DB test**

`tests/test_indexer_metadata.py`:

```python
import pytest
from unittest.mock import AsyncMock
from app.services.indexer import ProductIndexer


async def test_index_product_writes_category_metadata(truncate_kb):
    pool = truncate_kb
    embedding = AsyncMock()
    embedding.embed.return_value = [0.0] * 1024
    client = AsyncMock()
    client.get_for_rag.return_value = {
        "name": "Quần A", "brand": "X", "category": "Quần cầu lông",
        "specifications": [], "description": "mô tả",
    }
    indexer = ProductIndexer(pool, embedding, client)
    await indexer.index_product(101)
    async with pool.acquire() as conn:
        cat = await conn.fetchval(
            "SELECT metadata->>'category' FROM kb_chunks "
            "WHERE doc_type='product' AND source_id='101' LIMIT 1"
        )
    assert cat == "Quần cầu lông"
```

- [ ] **Step 2: Run to verify it fails**

Run: `DATABASE_URL=... uv run pytest tests/test_indexer_metadata.py -v`
Expected: FAIL (`cat is None` — metadata not written).

- [ ] **Step 3: Write category into metadata**

In `app/services/indexer.py`, change the INSERT inside `index_product` (lines 83-93) to include `metadata`:

```python
                import json

                category = product.get("category")
                metadata_json = json.dumps({"category": category} if category else {})
                for idx, chunk in enumerate(chunks):
                    embedding = await self._embedding.embed(chunk)
                    await conn.execute(
                        """
                        INSERT INTO kb_chunks
                            (doc_type, source_id, chunk_index, content, metadata, embedding)
                        VALUES ('product', $1, $2, $3, $4::jsonb, $5)
                        """,
                        source_id,
                        idx,
                        chunk,
                        metadata_json,
                        embedding,
                    )
```

(Move the `import json` to the top of the file alongside the other imports rather than inline if the repo style prefers it — check the existing import block.)

- [ ] **Step 4: Run to verify it passes**

Run: `DATABASE_URL=... uv run pytest tests/test_indexer_metadata.py -v`
Expected: PASS.

- [ ] **Step 5: Write the backfill script**

`scripts/backfill_chunk_metadata.py` — re-index every product so metadata is populated. Follow the pattern of any existing backfill script in `scripts/` (mirror its arg parsing / pool creation). Minimal form:

```python
"""Re-index all product chunks so kb_chunks.metadata->>'category' is populated.

Run once after deploying the metadata-writing indexer:
    uv run python -m scripts.backfill_chunk_metadata
"""

import asyncio
import logging

import httpx

from app.core.db import create_pool
from app.services.embedding import EmbeddingService
from app.services.indexer import ProductIndexer
from app.services.product_client import ProductClient

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("backfill_chunk_metadata")


async def main() -> None:
    pool = await create_pool()
    http = httpx.AsyncClient()
    indexer = ProductIndexer(pool, EmbeddingService(http), ProductClient(http))
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT DISTINCT source_id FROM kb_chunks WHERE doc_type='product'"
        )
    ids = sorted(int(r["source_id"]) for r in rows)
    log.info("re-indexing %d products", len(ids))
    for pid in ids:
        try:
            await indexer.index_product(pid)
        except Exception as exc:  # keep going; log the failures
            log.warning("product %s failed: %s", pid, exc)
    await http.aclose()
    await pool.close()
    log.info("done")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 6: Smoke-check the backfill imports**

Run: `uv run python -c "import scripts.backfill_chunk_metadata"` (Expected: no error). Full run is a deploy-time action (needs shop-api + Ollama up), not part of the test gate.

- [ ] **Step 7: Lint + commit**

Run: `uv run ruff check app/services/indexer.py scripts/backfill_chunk_metadata.py`
```bash
git add app/services/indexer.py scripts/backfill_chunk_metadata.py tests/test_indexer_metadata.py
git commit -m "feat(index): write category to chunk metadata + backfill script

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Response contract + chat orchestration rewrite (state → understand → filter → structured display_products)

**Files:**
- Modify: `app/models/schemas.py` (ChatResponse, lines 41-47)
- Modify: `app/routers/chat.py` (handler lines 26-97; add a structured display-products helper; stop using `_extract_recommended` for cards)
- Test: `tests/test_chat_router.py` (add cases)

**Interfaces:**
- Consumes: `ConversationStateStore` (Task 1, `app.state.conversation_state`), `QueryUnderstandingService` (Task 2, `app.state.query_understanding`), category-aware `RetrievalService.search` (Task 3), `ProductClient.get_pricing` (existing) for price sort, `settings.chat_display_products_max`.
- Produces: `ChatResponse` with new fields `intent`, `categories`, `display_products: list[int]`, `conversation_state: ConversationState`. New helper `_structured_display_products(chunks, tool_products, categories, cap) -> list[int]`.

- [ ] **Step 1: Extend the response schema**

In `app/models/schemas.py`, import `ConversationState` at the top:

```python
from app.services.conversation_state import ConversationState
```

Replace the `ChatResponse` class (lines 41-47) with:

```python
class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    # Legacy prose-scraped ids; kept for back-compat. Cards now use display_products.
    products: list[str] = Field(default_factory=list)
    order_draft: OrderDraft | None = None
    # Foundation additions:
    intent: str | None = None
    categories: list[str] = Field(default_factory=list)
    # The ONLY ids the frontend renders as cards for this message (structured, not scraped).
    display_products: list[int] = Field(default_factory=list)
    conversation_state: ConversationState = Field(default_factory=ConversationState)
```

- [ ] **Step 2: Write the failing unit test for the structured display-products helper**

Add to `tests/test_chat_router.py`:

```python
from app.routers.chat import _structured_display_products
from app.services.retrieval import Chunk


def _chunk(sid, cat="Quần cầu lông"):
    return Chunk(doc_type="product", source_id=sid, chunk_index=0,
                 content=f"Sản phẩm: P{sid}", distance=0.1)


def test_structured_display_products_from_chunks_capped_deduped():
    chunks = [_chunk("101"), _chunk("101"), _chunk("102"), _chunk("103"), _chunk("104"), _chunk("105")]
    ids = _structured_display_products(chunks, tool_products=[], cap=4)
    assert ids == [101, 102, 103, 104]  # retrieval order, deduped, capped, ints


def test_structured_display_products_prefers_tool_products():
    chunks = [_chunk("101")]
    tool_products = [{"id": "201", "name": "X"}, {"id": "202", "name": "Y"}]
    ids = _structured_display_products(chunks, tool_products=tool_products, cap=4)
    assert ids[:2] == [201, 202]  # recommend_similar_products results lead


def test_structured_display_products_ignores_non_numeric():
    ids = _structured_display_products([_chunk("abc"), _chunk("102")], tool_products=[], cap=4)
    assert ids == [102]
```

- [ ] **Step 3: Run to verify it fails**

Run: `uv run pytest tests/test_chat_router.py -k structured_display -v`
Expected: FAIL (`cannot import name '_structured_display_products'`).

- [ ] **Step 4: Implement the structured helper**

Add to `app/routers/chat.py` (near the other helpers):

```python
def _structured_display_products(
    chunks: list[Chunk], tool_products: list[dict], cap: int
) -> list[int]:
    """The product ids to render as cards, from STRUCTURED sources (not prose):
    recommend_similar_products results first, then retrieved product chunks, in
    order, deduped, numeric-only, capped. Replaces the substring scrape."""
    ordered: list[str] = [p["id"] for p in tool_products if p.get("id")]
    ordered += [c.source_id for c in chunks if c.doc_type == "product"]
    out: list[int] = []
    seen: set[int] = set()
    for sid in ordered:
        try:
            n = int(sid)
        except (TypeError, ValueError):
            continue
        if n > 0 and n not in seen:
            seen.add(n)
            out.append(n)
        if len(out) >= cap:
            break
    return out
```

- [ ] **Step 5: Run to verify it passes**

Run: `uv run pytest tests/test_chat_router.py -k structured_display -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Rewrite the handler**

Replace the body of `async def chat(...)` in `app/routers/chat.py` (lines 27-97) so the flow becomes: load state → analyze → filtered retrieval (using `qu.retrieval_query` embedding + `qu.categories`) → build catalog/context → tool loop (unchanged) → structured display_products → optional price sort → persist state → return. Concretely:

```python
@router.post("")
async def chat(request: ChatRequest, http_request: Request) -> ChatResponse:
    embedding_svc: EmbeddingService = http_request.app.state.embedding
    retrieval_svc: RetrievalService = http_request.app.state.retrieval
    llm_svc: LLMService = http_request.app.state.llm
    tool_dispatcher: ToolDispatcher = http_request.app.state.tool_dispatcher
    state_store = http_request.app.state.conversation_state
    qu_svc = http_request.app.state.query_understanding

    query = request.message.strip()
    if len(query) < settings.min_query_length:
        raise HTTPException(status_code=400, detail="Câu hỏi quá ngắn")

    with (
        propagate_attributes(session_id=request.session_id or None, trace_name="chat"),
        langfuse.start_as_current_observation(name="chat", as_type="span", input=query) as root,
    ):
        state = await state_store.load(request.session_id)
        qu = await qu_svc.analyze(query, state)

        with langfuse.start_as_current_observation(
            name="embed", as_type="embedding", input=qu.retrieval_query,
            model=settings.embedding_model,
        ):
            query_vec = await embedding_svc.embed(qu.retrieval_query)

        with langfuse.start_as_current_observation(
            name="retrieval", as_type="retriever", input=qu.retrieval_query
        ) as rspan:
            chunks = await retrieval_svc.search(
                query_vec, categories=qu.categories or None
            )
            rspan.update(output=[{"doc_type": c.doc_type, "source_id": c.source_id} for c in chunks])

        context = _format_context(chunks)
        catalog = _extract_product_catalog(chunks)
        system_content = SYSTEM_PROMPT.format(context=context)
        if catalog:
            listing = "\n".join(f"- {pid}: {name}" if name else f"- {pid}" for pid, name in catalog)
            system_content += (
                "\n\nDanh sách sản phẩm hợp lệ để gọi tool. Chọn ĐÚNG product_id ứng "
                "với TÊN sản phẩm khách hỏi; KHÔNG nhắc ID trong câu trả lời, KHÔNG "
                "dùng ID ngoài danh sách này:\n" + listing
            )
        else:
            system_content += (
                "\n\nNgữ cảnh không chứa sản phẩm nào. KHÔNG gọi tool với ID tự nghĩ "
                "ra; nếu khách hỏi giá hoặc tồn kho, trả lời rằng bạn không tìm thấy "
                "sản phẩm phù hợp."
            )

        messages: list[dict] = [{"role": "system", "content": system_content}]
        for m in request.chat_history:
            messages.append({"role": m.role, "content": m.content})
        messages.append({"role": "user", "content": query})

        answer, tool_products, order_draft = await _run_tool_loop(
            llm_svc, tool_dispatcher, messages
        )

        display = _structured_display_products(
            chunks, tool_products, settings.chat_display_products_max
        )
        if qu.price_preference == "cheapest" and display:
            display = await _sort_by_price(http_request.app.state.http, display)

        # Persist the updated conversation scope for the next (possibly elliptical) turn.
        state.categories = qu.categories or state.categories
        state.intent = qu.intent
        state.price_preference = qu.price_preference
        await state_store.save(request.session_id, state)

        root.update(output=answer)
        return ChatResponse(
            answer=answer,
            sources=_unique_sources(chunks),
            products=[str(i) for i in display],  # legacy mirror
            order_draft=order_draft,
            intent=qu.intent,
            categories=qu.categories,
            display_products=display,
            conversation_state=state,
        )
```

Add the price-sort helper (uses `ProductClient` directly on the shared http client; degrades on failure):

```python
async def _sort_by_price(http_client, ids: list[int]) -> list[int]:
    """Sort candidate ids ascending by live price (shop-api). On any failure keep
    the original order — never empty the list because pricing was unavailable."""
    from app.services.product_client import ProductClient

    client = ProductClient(http_client)
    priced: list[tuple[float, int]] = []
    for pid in ids:
        try:
            data = await client.get_pricing(pid)
            variants = data.get("variants") or []
            prices = [v.get("price") for v in variants if v.get("price") is not None]
            priced.append((min(prices) if prices else float("inf"), pid))
        except Exception:  # noqa: BLE001 — degrade gracefully
            priced.append((float("inf"), pid))
    priced.sort(key=lambda t: t[0])
    return [pid for _, pid in priced]
```

Remove the now-unused `_extract_recommended` call at old line 89. Keep the `_extract_recommended` function itself only if other code references it; otherwise delete it and its helpers (`_name_core`, `_NAME_PREFIXES`) if they become orphaned by THIS change (verify with a grep before deleting — do not remove pre-existing usage elsewhere).

- [ ] **Step 7: Update the tool-loop import test surface + run the full chat test file**

Run: `uv run pytest tests/test_chat_router.py -v`
Expected: PASS (existing `_parse_order_draft` / `_run_tool_loop` tests unchanged + the 3 new structured-display tests). If `_extract_recommended` was deleted, remove any test that imported it.

- [ ] **Step 8: Full suite + lint**

Run: `DATABASE_URL=... uv run pytest tests/ -q && uv run ruff check app/ && uv run ruff format --check app/routers/chat.py app/models/schemas.py`
Expected: all green, lint clean.

- [ ] **Step 9: Commit**

```bash
git add app/routers/chat.py app/models/schemas.py tests/test_chat_router.py
git commit -m "feat(chat): state-aware, category-filtered pipeline + structured display_products

Load session state -> understand query -> category-filtered retrieval -> tool loop
-> structured display_products (replaces prose scrape) -> persist state. Closes P1/P2/P5.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: shop-ui — response types + render `display_products` only

**Files:** (branch `feat/chatbot-foundation-ui` off shop-ui `origin/main` = `dae04d2`)
- Modify: `components/chatbot/types.ts` (ChatResponse + ChatMessage)
- Modify: `components/chatbot/chat-panel.tsx` (append + render)

**Interfaces:**
- Consumes: RAG `ChatResponse` with `display_products: number[]`, `intent`, `categories`, `conversation_state`.
- Produces: cards rendered strictly from `message.display_products`; the raw-sources fallback removed.

- [ ] **Step 1: Create the UI branch**

```bash
git -C /home/mgriffe-work/Desktop/TTTN/goodminton-shop-ui fetch origin
git -C /home/mgriffe-work/Desktop/TTTN/goodminton-shop-ui checkout -b feat/chatbot-foundation-ui origin/main
```

- [ ] **Step 2: Extend the types**

In `components/chatbot/types.ts`, add to `ChatResponse`:

```ts
export interface ChatResponse {
  answer: string;
  sources: SourceRef[];
  products?: string[];
  order_draft?: OrderDraft;
  /** Structured ids to render as cards for THIS message (supersedes products/sources). */
  display_products?: number[];
  intent?: string | null;
  categories?: string[];
}
```

And to `ChatMessage`:

```ts
  /** Structured card ids for this assistant message (backend-resolved). */
  display_products?: number[];
```

- [ ] **Step 3: Persist `display_products` on the assistant message**

In `components/chatbot/chat-panel.tsx`, in the `setMessages` append after `sendChat` (around lines 105-115), add `display_products: res.display_products,` to the pushed assistant object.

- [ ] **Step 4: Render only `display_products`; remove the fallback**

Replace the card-id computation in `MessageBubble` (origin/main lines 293-307) with:

```tsx
  const productIds = isUser
    ? []
    : Array.from(new Set(message.display_products ?? []))
        .filter((n) => Number.isInteger(n) && n > 0)
        .slice(0, 4);
```

Delete the now-unused `recommended` / `fromSources` / `rawIds` locals (they were the scrape-era fallback). Leave `ProductSourceCards` and the per-message binding untouched.

- [ ] **Step 5: Type gate**

Run: `cd /home/mgriffe-work/Desktop/TTTN/goodminton-shop-ui && npx tsc --noEmit`
Expected: exit 0, no output. (If `message.sources`/`message.products` become unused elsewhere, leave them — other code and the legacy fields still reference them.)

- [ ] **Step 6: Commit**

```bash
git add components/chatbot/types.ts components/chatbot/chat-panel.tsx
git commit -m "feat(chatbot): render cards from backend display_products, drop scrape fallback

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: shop-ui — fresh conversation per reload

**Files:**
- Modify: `components/chatbot/session.ts`
- Modify: `components/chatbot/chat-panel.tsx` (do not restore persisted history on mount)

**Interfaces:**
- Produces: a `session_id` that is regenerated on every page load (in-memory, not persisted), and an empty transcript on reload.

- [ ] **Step 1: Make the session id per-page-load**

Replace `components/chatbot/session.ts` so the id lives only in module memory (regenerated each full page load) instead of `localStorage`:

```ts
// Per-page-load chat session id. A reload starts a brand-new conversation, so
// server-side state (keyed by this id) and the visible transcript reset together.
let currentId: string | null = null;

export function getChatSessionId(): string {
  if (globalThis.window === undefined) return "";
  if (currentId) return currentId;
  currentId = globalThis.crypto?.randomUUID?.() ?? fallbackId();
  return currentId;
}

export function resetChatSessionId(): void {
  currentId = null;
}

function fallbackId(): string {
  const g = globalThis.crypto;
  if (g?.getRandomValues) {
    const buf = new Uint32Array(1);
    g.getRandomValues(buf);
    return `s-${buf[0].toString(36)}`;
  }
  return `s-${Date.now().toString(36)}`;
}
```

- [ ] **Step 2: Do not replay persisted history on reload**

In `components/chatbot/chat-panel.tsx`, the mount effect currently restores `messages` from `localStorage` (origin/main ~lines 43-49). Change it so a reloaded page starts with an empty transcript: remove the restore-from-`localStorage` on mount (keep the write effect if other features rely on it, but do not seed `messages` from it). The chat opens empty each page load, matching the fresh `session_id`.

- [ ] **Step 3: Type gate**

Run: `cd /home/mgriffe-work/Desktop/TTTN/goodminton-shop-ui && npx tsc --noEmit`
Expected: exit 0.

- [ ] **Step 4: Commit**

```bash
git add components/chatbot/session.ts components/chatbot/chat-panel.tsx
git commit -m "feat(chatbot): fresh conversation per page reload (ephemeral session id)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Manual end-to-end verification (after all tasks; needs full stack up)

With embed-independent stack up (infra + RAG on `feat/chatbot-foundation` + shop-ui on `feat/chatbot-foundation-ui` + `npm run dev`), and after running `uv run python -m scripts.backfill_chunk_metadata`:

1. **P1:** "mua quần" → cards are all pants; "cho tôi quần và giày" → both categories appear.
2. **P2:** "mua quần" then "rẻ nhất" → answer + cards scoped to the cheapest pants (not shirts / not whole catalog).
3. **P5:** the cards under each answer are exactly that turn's `display_products`; no stale/mismatched category.
4. **Regression:** a plain non-product question still answers; Redis stopped → chat still responds (stateless), no 500.
5. **Session:** reload the page → the chat is empty and a new `session_id` is issued (prior scope does not leak in).

---

## Self-Review (checklist run against the spec)

1. **Spec coverage:** §3 pipeline → Task 5; §4 state model → Task 1; §5 contract → Tasks 5 (RAG) + 6 (UI); §6 session lifetime → Task 7; §7 category source/metadata + backfill → Task 4; §8 hybrid extraction → Task 2; §9 error handling → Task 1 (Redis degrade), Task 2 (LLM-fallback + inherit), Task 5 (`_sort_by_price` degrade); §10 testing → each task's tests + the manual E2E; §11 non-goals honored (reserved fields only, no order SM / sanitize / image-search); §12 price source → decision (a) implemented in Task 5 `_sort_by_price`.
2. **Placeholder scan:** no TBD/TODO; every code step has concrete code; the one grounding step (Task 2 Step 1: discover real category strings) is explicitly a verify-and-adjust step with a default provided.
3. **Type consistency:** `ConversationState` fields consistent across Tasks 1/5; `search(..., categories=...)` signature matches between Task 3 (def) and Task 5 (call); `display_products: list[int]` (RAG) ↔ `number[]` (UI); `_structured_display_products(chunks, tool_products, cap)` signature consistent Task 5 def↔call; `QueryUnderstanding.retrieval_query`/`categories`/`price_preference` consistent Task 2↔5.
