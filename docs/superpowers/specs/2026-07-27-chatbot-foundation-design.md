# Chatbot Foundation — Design Spec

> **Scope:** Spec 1 of 3 in the sales-chatbot fix package. Closes **P1, P2, P5** from
> the audit `docs/superpowers/audits/2026-07-27-chatbot-sales-flow-audit.md`.
> Spec 2 (order state machine, P3) and Spec 3 (tool-result sanitize, P4) are separate,
> follow-on specs and are **out of scope here** (only a reserved field is added for Spec 2).

**Date:** 2026-07-27
**Branch:** `feat/chatbot-foundation` (RAG) + a matching UI branch, both off `origin/main`.
**Repos touched:** `goodminton-rag-service` (primary), `goodminton-shop-ui` (response contract + session lifetime).

---

## 1. Problem (from the verified audit)

The `/chat` endpoint is stateless and context-blind: each turn embeds only the latest
message and runs **one** cosine search over all of `kb_chunks` with **no category filter**
(`retrieval.py:30-40`), then scrapes product cards back out of the answer text by substring
match (`chat.py` `_extract_recommended`, `answer.lower().find(core)`). Consequences:

- **P1** — "mua quần" returns shirts; "quần + giày" returns only pants; unrelated categories leak in. Root: no intent/category extraction, no category filter, one blended query vector, cosine-only ranking.
- **P2** — "Mua quần?" then "Rẻ nhất?" loses scope. Root: `chat_history` feeds the LLM (`chat.py:82`) but **never** the retrieval query (`chat.py:33/49/54`); no server-side state.
- **P5** — text says pants, cards show shirts. Root: cards are scraped from prose (decoupled from the answer), plus a frontend fallback to raw retrieval sources.

The design does **not** fix these with prompt edits — they are architectural gaps in the
retrieval pipeline, backend orchestration, and the response contract.

## 2. Approach (chosen: A — deterministic query-understanding pipeline)

Insert a deterministic "understand the query" stage **before** retrieval, keep the existing
LLM + tool loop for generation, and replace prose-scraping with a structured
`display_products` contract. Category selection is **never** delegated to the 3B model
(its weak tool-calling is the root of P4), it is computed deterministically.

Rejected alternative **B (agentic tool-driven retrieval)**: give the LLM a
`search_products(category, …)` tool and let it choose filters. Rejected because it bets on
the exact weakness of the 3B model and is hard to test deterministically.

## 3. New pipeline order (`app/routers/chat.py`)

1. **Load state** — read `ConversationState` from Redis by `session_id`; fresh state if absent.
2. **Query understanding** (new `app/services/query_understanding.py`):
   - Hybrid extract `intent` + `categories` from the latest message (rule-based first, LLM fallback).
   - **Contextualize:** if the message names no category (elliptical, e.g. "rẻ nhất"), inherit `categories` from state.
   - Update `price_preference` (e.g. "rẻ / rẻ nhất / giá thấp" → `cheapest`).
   - Produce a contextualized `retrieval_query` string.
3. **Category-filtered retrieval** (`app/services/retrieval.py`):
   - Add optional `categories: list[str] | None` and `doc_type` filter params.
   - `WHERE metadata->>'category' = ANY($categories)` (plus existing cosine order).
   - **Multi-category:** retrieve per category then merge with a per-category quota so
     "quần + giày" returns both (not a single global top-k that one category dominates).
4. **Generation** — unchanged LLM + tool loop. System prompt receives the **category-scoped**
   catalog via the existing `_extract_product_catalog` (id: name pairs from PR #2).
5. **`display_products`** — the backend returns the resolved, category-filtered candidate set
   **bound to this message**, built from structured ids + tool results. **Remove** the
   substring scrape (`_extract_recommended`) and its call site. Cap the list at a new config
   `chat_display_products_max` (default 4, matching the frontend card cap).
6. **Price ordering** — when `price_preference == "cheapest"`, fetch live prices for the
   candidate ids via `product_client` (source of truth = shop-api) and sort ascending before
   capping. (Decision (a): live prices, never the stale in-description numbers.)
7. **Persist state** — write updated `ConversationState` to Redis with a TTL.

## 4. Conversation state model

`ConversationState` (serialized JSON, Redis key `chat:state:{session_id}`, TTL ~30–60 min):

```
intent:             str | None          # browse | price | stock | buy  (coarse)
categories:         list[str]           # e.g. ["pants"] or ["pants","shoes"]
price_preference:   str | None          # cheapest | None  (extensible later)
selected_product_id: int | None         # RESERVED for Spec 2 (order flow); unused here
order_status:       str | None          # RESERVED for Spec 2; unused here
```

- Keyed by `session_id`, which the frontend already sends (`ChatRequest.session_id`).
- **Backend owns state** — the client sends no state fields inbound beyond `session_id`.
- Redis is already in the infra (`docker-compose.infra.yml`); reuse it.

## 5. Response contract (`app/models/schemas.py` + `goodminton-shop-ui`)

`ChatResponse` gains (additive, non-breaking):

```
intent:            str | None
categories:        list[str]
display_products:  list[int]      # the ONLY ids the frontend renders as cards for this message
conversation_state: object        # the ConversationState above (for display/debug)
```

`answer`, `sources`, `products`, `order_draft` stay for now. **`display_products` supersedes
`products` for card rendering**; `products` is left in place so Spec 2/3 can retire it cleanly.

**Frontend (`goodminton-shop-ui/components/chatbot`):**
- Render cards from **`display_products` of that exact message only**. Keep the existing
  per-message immutable binding (`chat-panel.tsx` — already correct, do not change its shape).
- **Remove** the raw-sources fallback that renders `sources.slice(0,3)` when `products` is empty.
- Add `display_products?`, `intent?`, `categories?`, `conversation_state?` to the `ChatResponse` type.

## 6. Session lifetime — fresh conversation per reload

Decision: **every page reload starts a clean conversation.**
- `components/chatbot/session.ts`: generate the `session_id` fresh on mount (in-memory /
  regenerated per page load) instead of persisting it to `localStorage`.
- The displayed chat history starts empty on reload (do not replay persisted history for a
  reloaded page). This keeps the server-side state (keyed by the per-load `session_id`) and the
  visible transcript in sync — a reloaded tab is a new session with empty Redis state.

## 7. Category data source

The product `category` is **already available at index time** — `indexer.py:46` reads
`product['category']` to build the "Danh mục: …" text line. The only change is to **also write
it into the `kb_chunks.metadata` JSONB column** (currently left empty per audit) so it is a
queryable field, not just free text inside the embedded content.

- Modify the indexer to populate `metadata = {"category": <category>}` (extend, don't replace,
  if other keys are later added).
- **One-time backfill / re-index** of existing `kb_chunks` product rows so `metadata->>'category'`
  is populated for already-indexed products. (A reindex script analogous to the existing
  backfill utilities.)

## 8. Extraction engine (hybrid)

`app/services/query_understanding.py`:
- **Rule-based first:** a keyword→category map covering the store's real categories
  (vợt / giày / quần / áo / phụ kiện + common synonyms and singular/plural/diacritic variants),
  plus a coarse intent map (browse / price / stock / buy) and a `price_preference` map
  ("rẻ", "rẻ nhất", "giá thấp", "rẻ hơn" → `cheapest`).
- **LLM fallback:** only when the rules extract nothing, ask the LLM to classify
  category+intent (kept small and optional so 3B latency/unreliability is not on the hot path).
- The category vocabulary must be sourced from shop-api's real category list (the same values
  that appear in `product['category']` at index time) — no invented labels.

## 9. Error handling / degradation

- **Redis unavailable** → degrade to today's stateless behavior (no state load/persist), **not** a 500.
- **Extraction finds no category** → run retrieval with **no** category filter (today's behavior),
  never return an empty result set because a guess failed.
- Keep the **no-distance-threshold** philosophy (do not add a cosine cutoff that could empty results).
- Price fetch failure (shop-api down) → skip price sorting, keep the unsorted candidate set.

## 10. Testing

RAG (pytest):
- Extraction: "mua quần" → `["pants"]`; "quần và giày" → `["pants","shoes"]`; "rẻ nhất" alone →
  inherits state categories; unknown phrasing → LLM fallback path.
- Retrieval filter: `WHERE` restricts to category; multi-category quota returns **both** categories;
  no-category call preserves current behavior.
- Contextualization: "mua quần" then "rẻ nhất" → retrieval query scoped to pants + `price_preference=cheapest`.
- `display_products`: built from structured ids (not scraped); equals the filtered/ranked set; capped.
- Price ordering: `cheapest` sorts by live `product_client` price; shop-api-down path skips sort gracefully.
- State: Redis roundtrip (save→load), TTL set; Redis-down degradation returns a valid response.

shop-ui:
- `npx tsc --noEmit` exit 0 (no test runner).
- Cards render only `display_products`; raw-sources fallback removed; per-message binding intact.

## 11. Non-goals (scope guard)

- **Order state machine (P3)** — Spec 2. Only `selected_product_id` / `order_status` fields are
  reserved in the state model; no state-machine logic here.
- **Tool-result sanitize / raw-JSON leak (P4)** — Spec 3.
- Do not touch the image-search path (`feat/image-search`), `search.py`, or `image_*`.
- Keep `prepare_order` read-only (price + stock only).
- Do not restructure the frontend's per-message binding mechanism (it is already correct).

## 12. Open decision left to confirm

- **Price source for `price_preference`** — spec assumes **(a)** live `product_client` fetch +
  backend sort. Alternative **(b)** store price in `metadata` JSONB at index time (faster, but
  price is volatile and would go stale). Change here only affects step 6 + the indexer.
