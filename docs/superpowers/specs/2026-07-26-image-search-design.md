# Image Search — Design Spec

> Status: design draft for review (2026-07-26). Authored via ultracode workflow (ground → design → adversarial critique). Next: user review → `writing-plans`.
> Cross-repo + a NEW service: `goodminton-image-embed-service` (new), `goodminton-rag-service`, `goodminton-shop-api`, `goodminton-shop-ui`.

## Goal

Let a shopper find products by an **image**: upload/paste a photo → get visually-similar products. Two entry points (storefront search bar 📷 and the chatbot) both hit the same backend. Existing text search is untouched.

## Architecture (grounded + review-hardened)

SigLIP image embeddings + pgvector cosine, mirroring the codebase's existing vector patterns (similar-products / hybrid recs). The **matching logic lives in RAG**; a **separate embedding service** hosts the model; **shop-api** owns the schema + catalog hydration; the **UI** reuses the existing results grid + chatbot cards.

```
Index (backfill + sync): shop-api image URLs (Cloudinary) → RAG downloads → embed-service (SigLIP) → product_image_embeddings
Search: image → RAG /search/image → embed-service (vector) → pgvector cosine → product_ids
        → hydrate via shop-api → storefront grid (entry 1) / chatbot cards (entry 2)
```

## Tech Stack
- **embed-service** (new): FastAPI + `transformers` SigLIP `google/siglip-base-patch16-224` (768-dim) + Pillow, **CPU-only torch**. Runs on port **8001**.
- RAG: FastAPI, asyncpg/pgvector, httpx. shop-api: Spring Boot + Flyway. UI: Next.js 16.

---

## Global Constraints & Decisions (every task inherits these)

**Architecture decisions** (chosen; each overridable — see "Simplification options"):
1. **Separate embed-service** hosts SigLIP (isolates torch). RAG + indexer call it over HTTP. *(User choice.)*
2. **Index ALL gallery images** per product (`resources` rows under `PRODUCT_THUMBNAIL`), not just the thumbnail. Query aggregates per product via **plain `GROUP BY … MIN(embedding <=> $1)`** (HNSW-CTE optimization deferred to future scale). *(Honors "production đầy đủ".)*
3. **Both entry points**: storefront search-bar 📷 **and** chatbot paste-image. *(User choice.)*
4. **Incremental freshness via RabbitMQ** (shop-api publishes an image-changed event; RAG consumer re-indexes) **plus** a backfill script for initial load. Ship the shop-api + RAG halves **together**. *(Honors "production đầy đủ".)*
5. Model `google/siglip-base-patch16-224`, 768-dim, L2-normalized, CPU. **Text search stays unchanged** (shop-api FTS).

**Hardening requirements (from adversarial review — non-negotiable):**
- **H1 — one response contract:** RAG `/search/image` returns `{"product_ids": ["42","17",…]}` — **strings, ranked by ascending distance**. Both entry points read `product_ids`. (No `results`/object-array drift.)
- **H2 — one multipart field name `file`** across all hops: FE→RAG uses `file`; RAG→embed-service uses `file`. (Matches existing `productsApi.uploadImage`.)
- **H3 — no distance threshold initially:** return top-K regardless of distance (SigLIP is sigmoid-trained; a guessed cutoff silently returns empty = "looks broken"). Log distances; add a calibrated threshold only after measuring real distributions.
- **H4 — decompression-bomb guard:** set `Image.MAX_IMAGE_PIXELS` and catch `DecompressionBombError` in the embed-service decode path. Byte caps (RAG 8 MB outer, embed 10 MB) are **not** sufficient alone.
- **H5 — rate-limit** `/search/image` (unauthenticated + CPU-bound): a crude per-IP limit in RAG.
- **H6 — embed-service is host-bound to `127.0.0.1:8001`** and exposes ONLY `/embed/image` + `/health`. **No URL-fetch endpoint** (`/embed/images` removed) → **no SSRF surface**. RAG downloads Cloudinary images itself (trusted source) and POSTs bytes.
- **H7 — graceful 503:** if the embed-service is unreachable / model not loaded, RAG `/search/image` returns a friendly **503** ("tìm ảnh đang khởi động, thử lại"), never a 500 stacktrace. RAG `/health` does **not** depend on embed-service.
- **H8 — `is_visible` filtering + over-fetch:** RAG returns ~2–3× `top_k` ids; hydration keeps the first `top_k` **visible** products. `list-items` filters `is_visible` server-side.
- **H9 — resilient indexing:** per-image download+embed wrapped in try/except; skip failures (log); atomic replace over whatever succeeded; **if ALL images fail, do NOT wipe existing rows** (skip replace). Cap decoded size on this path too.
- **H10 — score sanity:** either drop `score` from the response (ranking = distance is enough) or clamp `similarity = max(0, 1 - distance)`.
- **H11 — build-time internet:** the SigLIP model is baked into the embed-service image at build → `docker build` needs internet once; document (or `docker save/load` for offline defense).
- **H12 — memory: 3g** on the embed-service container (SigLIP RSS ~1.5–2.5 GB); `TORCH_NUM_THREADS` = container cpus.

---

## Component Design

### 1. `goodminton-image-embed-service` (NEW)
- FastAPI app, model loaded once at startup (`SiglipModel` + `SiglipImageProcessor`, `use_safetensors=True`), `torch.set_num_threads(cpus)`.
- `POST /embed/image` (multipart field **`file`**) → decode with pixel-bomb guard (H4) → `get_image_features` → **L2-normalize** → `{"embedding": [..768..]}`.
- `GET /health` → `{status, model_loaded}`.
- **No URL-fetch endpoint** (H6).
- Deps: CPU-only torch (pinned via the PyTorch CPU index — keeps the image from ballooning with CUDA libs), transformers, pillow, fastapi, uvicorn, python-multipart.
- Dockerfile: uv-based (mirror RAG's), **bake the model at build** (H11). `EXPOSE 8001`.
- Compose: new service, bound `127.0.0.1:8001:8001` (H6), `memory: 3g` (H12), `restart: unless-stopped`, healthcheck on `/health` with a generous `start_period`.

### 2. RAG (`goodminton-rag-service`)
- **`EmbedClient`** (mirror `product_client.py`): `embed_image(bytes) -> list[float]`, base URL `settings.embed_service_url` (`http://localhost:8001`), POSTs multipart field `file` (H2), timeout; on connect error / non-200 → raise a typed error the router maps to 503 (H7).
- **`ImageSearchService`**: embed the query → cosine search:
  ```sql
  SELECT product_id, MIN(embedding <=> $1) AS distance
  FROM product_image_embeddings
  GROUP BY product_id
  ORDER BY distance ASC
  LIMIT $2            -- $2 = over_fetch = top_k * 3  (H8)
  ```
  Return product_ids as strings, ranked (H1, H3 = no threshold).
- **`POST /search/image`** router (multipart `file`): validate content-type + byte cap → `EmbedClient` → `ImageSearchService` → `{"product_ids": [...]}`. Rate-limited (H5); embed-service failure → 503 (H7).
- **`ImageIndexer`**: for a product id → GET its image URLs from shop-api internal → **download each (resilient, H9)** → POST bytes to embed-service → **atomic replace** rows in `product_image_embeddings` for that product (skip-on-all-fail, H9). `delete_product_images(product_id)`.
- **Consumer hook**: extend `product_consumer.py` to also re-index images when the event's `fieldsChanged` includes the new image marker (see shop-api §4). Ships with the shop-api half (decision #4).
- **Backfill script** (`scripts/backfill_product_images.py`): reuse `scripts/backfill_products.py`'s `SELECT id FROM products WHERE is_visible=true` (not the kb_chunks hack) → index each product's images.
- **Config**: `embed_service_url` (`http://localhost:8001`), `image_search_top_k`, `image_search_over_fetch_factor=3`, `image_max_upload_bytes`. Router wired in `main.py`.

### 3. shop-api (`goodminton-shop-api`)
- **Flyway migration** `product_image_embeddings(product_id int, resource_id int, url text, embedding vector(768), PRIMARY KEY(resource_id))` + HNSW cosine index; FK on `product_id → products(id)` ON DELETE CASCADE (keep the cascade cleanup; **drop the `resource_id` FK** to avoid write-time races — H per review #11). `CREATE EXTENSION vector` already exists (V7).
- **Internal image endpoint** `GET /api/internal/products/{id}/images` (X-Internal-Key) → `[{resourceId, url, sortOrder}]` via existing `resourceService.listByOwner(PRODUCT_THUMBNAIL, id)`.
- **Hydration endpoint** `GET /api/products/list-items?ids=1,2,3` → `List<ProductListItemResponse>` (reuse `toListItemResponse`), **preserve requested order**, **filter `is_visible`** (H8), public (guests use image search), batch the thumbnail lookup (avoid N+1).
- **Image-changed event**: publish `ProductChangedEvent.updated(productId, Set.of("images"))` from `uploadProductImage` + `deleteProductImage` (currently they publish nothing — the freshness gap). Routed via the existing `product.*` binding. (decision #4)

### 4. UI (`goodminton-shop-ui`)
- **searchApi additions**: `searchByImage(file) → {product_ids}` (POSTs multipart `file` to `NEXT_PUBLIC_RAG_API_URL/search/image`); `listItemsByIds(ids) → ProductListItemResponse[]` (shop-api).
- **Entry 1 — search bar** (`header-search.tsx`): a 📷 button + hidden `file` input (`accept="image/jpeg,image/png,image/webp"`, client-side size check + downscale to keep < a few MB — H per review #8) → `searchByImage` → hydrate → show in the `/products` grid in an "image results" mode (result set carried to the page; `ProductGrid` unchanged). Loading + empty ("Không tìm thấy sản phẩm giống ảnh") states.
- **Entry 2 — chatbot** (chat input + `chat-panel.tsx`): image attach button + `onPaste` handler + preview → POST directly to `/search/image` → render returned products as **product cards** (reuse `ProductSourceCards`) with a line "Đây là các sản phẩm giống ảnh của bạn". The chat LLM is bypassed (text-only, can't see images).
- **Verification:** repo has NO test runner → gate is `npx tsc --noEmit` + concrete manual checks.

## Data flow / contracts
- Query response: `{"product_ids": ["42","17",…]}` (H1). Multipart field `file` everywhere (H2).
- Hydration: RAG ids → shop-api `list-items?ids=` (visible-only, ordered) → grid/cards.
- Index: shop-api `/api/internal/products/{id}/images` → RAG downloads → embed-service `/embed/image` → `product_image_embeddings`.

## Security / correctness (requirements, not optional)
All H1–H12 above. Summary of the load-bearing ones: pixel-bomb guard (H4), rate-limit (H5), embed-service `127.0.0.1`-only with no URL-fetch (H6), graceful 503 (H7), `is_visible` filter (H8), resilient indexing (H9), no premature threshold (H3).

## Scope

**In scope:** embed-service; RAG image search + indexer + backfill + consumer hook; shop-api migration + internal image endpoint + list-items + image event; UI both entry points; all hardening.

**Out of scope (YAGNI):** vision LLM (bot *describing* images), blended text+image queries (SigLIP text query is a fast-follow), variant-level images, HNSW two-stage CTE (until catalog is large), a calibrated distance threshold (add after measuring — H3).

**Simplification options (if you want a leaner MVP — pick at review):**
- **S1 Thumbnail-only** instead of gallery → removes the `GROUP BY MIN` aggregation (plain top-k), ~5–10× less embed volume. Upgrade to gallery later is purely additive (schema already keyed by `resource_id`).
- **S2 Backfill-only** (drop the incremental RabbitMQ sync) → removes the fragile cross-repo image-event seam; re-run backfill after adding products. Safe if images are static during the demo.
- **S3 Chatbot-only** (drop the storefront 📷) → removes the shop-api `list-items` endpoint + `SecurityConfig` change + the `/products` image-mode; chatbot reuses `ProductSourceCards` with zero new shop-api surface.
These are the review's recommended MVP cuts; the current spec is the fuller "production" version.

## Testing
- **embed-service:** unit test `/embed/image` returns 768-dim normalized vector for a fixture image; pixel-bomb rejection; bad-content-type 400.
- **RAG:** `ImageSearchService` ranking + per-product MIN aggregation + over-fetch (DB fixture with known vectors); `EmbedClient` error → 503 mapping; `ImageIndexer` resilient (one image fails → others still indexed; all fail → no wipe); backfill id source.
- **shop-api:** migration + `product_image_embeddings`; internal image endpoint (X-Internal-Key); `list-items` order + `is_visible` filter; image-changed event published on upload/delete.
- **UI:** `npx tsc --noEmit` clean; manual: 📷 upload → grid; chatbot paste → cards; empty/error states.

## Build / deploy notes
- embed-service image baked with the model (H11) → build needs internet once. `memory: 3g` (H12). Bound `127.0.0.1:8001`.
- RAG env: `EMBED_SERVICE_URL=http://localhost:8001`.
- Backfill is a one-off job after deploy (and after bulk product/image imports).
- Prod box already runs Ollama 14B + RAG + infra — confirm RAM headroom for +SigLIP before committing.

## Open items for reviewer
- Confirm the 4 architecture decisions vs the S1–S3 simplifications (esp. gallery-vs-thumbnail and sync-vs-backfill — the review leaned leaner; this spec leaned "production").
- Confirm the embed-service lives as a new sibling repo/folder `goodminton-image-embed-service`.
- SigLIP vs a calibrated threshold: ship no-threshold first (H3), revisit after measuring.
