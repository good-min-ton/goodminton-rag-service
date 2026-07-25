# Chatbot Order Assistant — Design Spec

> Status: approved design (2026-07-26). Next step: `writing-plans`.
> Cross-repo feature: `goodminton-rag-service` (RAG) + `goodminton-shop-ui` (UI). **`goodminton-shop-api` gets ZERO changes.**

## Goal

Let a shopper place an order from the chatbot: the RAG assistant understands the buy request, resolves the exact product variant, assembles a **priced, stock-checked order draft**, and returns it in the chat response; the frontend renders a confirmation card and places the order via the **existing** checkout call under the user's own JWT.

## Architecture (grounded, security-reviewed)

Two-part split, chosen because the shop-api order endpoint requires a `CUSTOMER` JWT and the RAG service holds only a **read-only** internal API key:

1. **RAG stays read-only.** A new tool `prepare_order` reads pricing + inventory via the existing `/api/internal/*` endpoints, computes a draft, and returns it as a new `order_draft` field on the chat response. RAG never writes an order and never handles the user's JWT.
2. **Frontend executes the write.** A new `OrderConfirmCard` renders the draft, collects the shipping address, and calls the **existing** `ordersApi.create` with the logged-in user's Bearer token (auto-attached). The user's click is the single write gate.

Red-team confirmed this closes cross-user ordering, privilege escalation, and price-falsification: buyer identity comes from the JWT (server-side `currentAccountProvider.getCurrentAccount()`), the order request carries no price/customer/store fields (server re-prices from `ProductVariant` and resolves the store), and there is no RAG→`POST /api/orders` path.

## Tech Stack

- RAG: FastAPI, hand-rolled Ollama tool-calling (not LangChain), `httpx` → shop-api internal endpoints, Pydantic v2. Local LLM `qwen2.5:3b-instruct-q4_K_M` (config-swappable to 14B).
- UI: Next.js 16, React 19, Zustand auth store, `@tanstack/react-query`, existing `ordersApi`/`api.ts` Bearer plumbing.

---

## Global Constraints (every task inherits these — do not violate)

1. **shop-api is not modified.** No new endpoints, no schema changes.
2. **RAG never writes an order.** `prepare_order` performs only GET reads (`get_pricing`, `check_inventory`). The write is exclusively frontend-side via the existing `ordersApi.create`.
3. **The order request carries no price/customer/store.** `CreateOrderRequest` stays `{ items:[{variantId, quantity}], recipientName, recipientPhone, recipientAddress, recipientEmail?, note?, paymentMethod }`. Draft prices are display-only; the server re-prices.
4. **The chat REQUEST is unchanged.** No token, no user id sent to RAG. The JWT never leaves the browser toward RAG. Only the chat RESPONSE gains `order_draft`.
5. **ONE canonical `order_draft` schema** (below) is byte-identical across the tool output, the Pydantic model, and the UI type. This is the #1 integration invariant.
6. **Payment is COD only** for this feature.
7. **Never silently coerce user intent.** Invalid quantity / unknown variant → a warning + dropped line, never a fabricated or clamped line.
8. **Any warning blocks the whole order.** The confirm button is disabled when `warnings.length > 0` OR `items.length === 0`.

---

## The canonical `order_draft` contract (single source of truth)

This exact shape is emitted by the `prepare_order` tool (JSON string), validated by the RAG Pydantic `OrderDraft` model, serialized on `ChatResponse.order_draft`, and typed identically in the UI. Field names and types are load-bearing.

```jsonc
// order_draft (null when the turn produced no draft)
{
  "items": [
    {
      "product_id": "12",       // string (RAG codebase convention); UI does Number() before ordersApi.create
      "variant_id": "45",       // string; UI does Number()
      "product_name": "Vợt Yonex Astrox 100ZZ",
      "size": "4U",             // string | null  — kept SEPARATE from name so the user can catch a wrong-variant map
      "color": "Đỏ",            // string | null
      "quantity": 2,            // int, >= 1 (invalid quantities never produce a line)
      "unit_price": 3200000.0,  // float VND, display-only (salePrice if present else price)
      "line_total": 6400000.0,  // float VND, display-only (unit_price * quantity)
      "in_stock": true          // central-store availability >= quantity
    }
  ],
  "total": 6400000.0,           // float VND, sum of line_total over included lines (display-only)
  "currency": "VND",
  "warnings": [                 // flat human-readable strings; ANY entry disables ordering
    "Vợt Yonex Astrox 100ZZ (Đỏ 4U) chỉ còn 1 tại kho, cần 2."
  ]
}
```

Deliberately **dropped** from the raw design variants (YAGNI): no `ok` boolean (derive from `warnings`/`items`), no typed issue `code` enum, no combined `name` string (split into `product_name`/`size`/`color`). Rationale: the UI only needs to (a) display a human warning and (b) decide whether to disable the button — flat strings do both.

---

## Component Design — RAG (`goodminton-rag-service`)

### R1. `prepare_order` tool — `app/services/tools.py`

**Schema** — new `TOOL_SCHEMAS` entry (after the `recommend_similar_products` entry). Flat scalar args only (3B handles this far better than nested objects); the one unavoidable nesting is the `items` array, capped at 20.

```python
{
    "type": "function",
    "function": {
        "name": "prepare_order",
        "description": (
            "Tạo ĐƠN HÀNG NHÁP (chưa đặt) đã tính giá + kiểm tra tồn kho để khách xác "
            "nhận trên giao diện. CHỈ gọi khi khách muốn mua/đặt VÀ đã biết variant_id "
            "(PHẢI gọi get_pricing trước để lấy variant_id). Tool này KHÔNG tạo đơn thật."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "maxItems": 20,
                    "description": "Danh sách dòng sản phẩm cần đặt.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "product_id": {"type": "integer", "description": "ID sản phẩm (từ danh sách hợp lệ)."},
                            "variant_id": {"type": "integer", "description": "ID variant (lấy từ get_pricing)."},
                            "quantity": {"type": "integer", "description": "Số lượng, >= 1."},
                        },
                        "required": ["product_id", "variant_id", "quantity"],
                    },
                }
            },
            "required": ["items"],
        },
    },
},
```

**Dispatch** — new branch in `ToolDispatcher.execute()` after the `check_inventory` branch, falling through to the shared `return json.dumps(result, ensure_ascii=False)` and the shared `try/except` error ladder (no new try/except):

```python
elif name == "prepare_order":
    result = await self._prepare_order(arguments.get("items") or [])
```

**`_prepare_order` method** — deterministic; the model supplies only three integers per line, every price/stock number is computed here from shop-api. **No new `ProductClient` method** — reuse the existing `get_pricing` and `check_inventory`.

Logic per line (cap `items` to the first 20):
1. Coerce `product_id`, `variant_id`, `quantity` to `int` (bad casts hit the shared `(KeyError, TypeError, ValueError)` handler).
2. **Reject, never clamp**, invalid quantity: `quantity <= 0` or `quantity > 20` → append a warning, skip the line (no `items` entry).
3. `get_pricing(product_id)` (cache per product_id to dedupe). Find the variant by `variantId == variant_id`. If not found → warning `"variant_id … không thuộc sản phẩm …, đã bỏ qua."`, skip the line.
4. `unit_price = salePrice if salePrice is not None else price` (null salePrice = no discount, never 0). `line_total = unit_price * quantity`.
5. `check_inventory(variant_id)` returns a per-store list `[{storeId, storeName, quantity}]`. Match the **central-store row by `storeName == settings.central_store_name`**; if no match, treat available as `0` (conservative — fail toward out-of-stock, never fail-open). `in_stock = central_qty >= quantity`; if not, append an out-of-stock warning (keep the line, `in_stock=false`).
6. Build the line with split `product_name`/`size`/`color` (from `productName`, `colorName`, `sizeName`).

Return `{"items": [...], "total": <sum>, "currency": "VND", "warnings": [...]}`.

Exact shop-api field names (verified): pricing variant = `variantId, colorName, sizeName, skuCode, price, salePrice`; inventory row = `storeId, storeName, quantity`.

### R2. Pydantic models — `app/models/schemas.py`

Near `SimilarProduct` (string-id convention: `product_id: str`, matching `SourceRef.source_id`):

```python
class OrderDraftItem(BaseModel):
    product_id: str
    variant_id: str
    product_name: str
    size: str | None = None
    color: str | None = None
    quantity: int
    unit_price: float
    line_total: float
    in_stock: bool

class OrderDraft(BaseModel):
    items: list[OrderDraftItem] = Field(default_factory=list)
    total: float = 0.0
    currency: str = "VND"
    warnings: list[str] = Field(default_factory=list)
```

Add one optional field to `ChatResponse` (mirrors the `products` default style):

```python
    order_draft: OrderDraft | None = None
```

`ChatRequest` is **unchanged**.

### R3. Threading `order_draft` — `app/routers/chat.py`

Mirror the `tool_products` path mechanically (a draft is a structured object the tool emits, present-or-absent — **not** inferred from answer prose like `_extract_recommended`). Edits:

1. `_run_tool_loop` return type → `tuple[str, list[dict], dict | None]`.
2. Local `order_draft: dict | None = None` beside `tool_products`.
3. Beside the existing collector, add to the **fresh-execution** branch:
   ```python
   if name == "recommend_similar_products":
       _collect_tool_products(result, tool_products)
   elif name == "prepare_order":
       parsed = _parse_order_draft(result)
       if parsed is not None:
           order_draft = parsed   # last successful prepare_order wins
   ```
4. Carry `order_draft` through all return sites (normal exit, forced-final-answer exit, max-iterations exit).
5. Handler: unpack the 3-tuple and pass `order_draft=order_draft` into `ChatResponse(...)` (Pydantic coerces the dict into `OrderDraft`).
6. New helper (mirrors `_collect_tool_products`):
   ```python
   def _parse_order_draft(result: str) -> dict | None:
       try:
           data = json.loads(result)
       except (ValueError, TypeError):
           return None
       if not isinstance(data, dict) or "items" not in data or "error" in data:
           return None   # centralized {"error": ...} tool payloads → no draft
       return data
   ```

**Repeat-guard hardening (red-team HIGH).** The buy flow needs ≥2 sequential tool turns (`get_pricing` → `prepare_order`). The global repeat guard (`MAX_REPEATED_CALLS = 2`) can force a tool-free final answer before `prepare_order` ever runs, leaving the model claiming a card that doesn't exist. Mitigation:
- Raise `MAX_REPEATED_CALLS` to `3` (small headroom for the 2-step flow; still bounded by `MAX_TOOL_ITERATIONS = 10`).
- Reword the forced-final-answer system message so that, if no order was prepared, the model says it could not finish preparing the order and asks the user to try again — it must NOT invent a confirmation card.

### R4. Prompt — `app/core/prompts.py`

Extend the existing "QUY TẮC BẮT BUỘC VỀ GIÁ VÀ TỒN KHO" block (reuses the `product_id` allow-list already injected in `chat.py`). Add, terse (3B follows short imperatives better):

```
- KHI KHÁCH MUỐN MUA / ĐẶT HÀNG:
  1) PHẢI gọi get_pricing(product_id) trước để lấy variant_id + size/màu.
  2) Chọn ĐÚNG MỘT variant_id khớp size/màu khách yêu cầu, rồi gọi
     prepare_order(items) với product_id, variant_id, quantity.
  3) product_id CHỈ lấy từ danh sách hợp lệ; variant_id CHỈ lấy từ get_pricing.
     TUYỆT ĐỐI không bịa product_id/variant_id.
- Nếu thiếu size/màu/số lượng: hỏi lại ĐÚNG MỘT câu gọn rồi mới gọi prepare_order.
- Sau khi prepare_order trả về: nhắc lại size/màu đã chọn cho khách kiểm tra, và mời
  khách BẤM XÁC NHẬN trên thẻ đơn hàng. KHÔNG nêu lại tổng tiền bằng chữ (để thẻ hiển thị).
  TUYỆT ĐỐI KHÔNG nói đơn đã được đặt/thành công — việc đặt do khách bấm xác nhận.
- KHÔNG hỏi địa chỉ giao hàng trong chat (thẻ đơn hàng sẽ thu địa chỉ).
- Nếu không có sản phẩm phù hợp trong ngữ cảnh: nói shop chưa có, KHÔNG gọi prepare_order.
```

The "nhắc lại size/màu" rule + split size/color on the card are the only defenses against a 3B wrong-variant map (no programmatic fix exists). The "KHÔNG nêu lại tổng tiền bằng chữ" rule removes prose-vs-card total mismatch.

### R5. Config — `app/core/config.py`

One new setting, with the **grounded** default from the seeded DB:

```python
central_store_name: str = "Goodminton HQ - Di An"   # Store.name where is_central=true (verified 2026-07-26)
```

Env-overridable (`CENTRAL_STORE_NAME`). This is the only way to identify the central inventory row (the endpoint exposes no `isCentral` flag). If wrong, every line reads available=0 and spuriously warns out-of-stock (fail-safe, not fail-open).

---

## Component Design — UI (`goodminton-shop-ui`)

### U1. Types — `components/chatbot/types.ts`

Add the `OrderDraft` shape mirroring the canonical contract **exactly** (string ids), and thread it onto the two existing interfaces the same way `products?: string[]` flows. **`ChatRequest` is not touched.**

```ts
export interface OrderDraftItem {
  product_id: string;
  variant_id: string;
  product_name: string;
  size: string | null;
  color: string | null;
  quantity: number;
  unit_price: number;
  line_total: number;
  in_stock: boolean;
}
export interface OrderDraft {
  items: OrderDraftItem[];
  total: number;
  currency: string;
  warnings: string[];
}
// ChatResponse += order_draft?: OrderDraft
// ChatMessage  += order_draft?: OrderDraft   (assistant-only)
// ChatMessage  += placedOrderId?: number     (client-only; set after a successful order)
```

### U2. Threading + durable single-write guard — `components/chatbot/chat-panel.tsx`

- In the `send` callback, spread `order_draft: res.order_draft` onto the new assistant message (parallels `products`). It persists to `localStorage["gm.chat-history"]` with the rest of the message.
- Mount the card inside `MessageBubble`, sibling to `ProductSourceCards`, gated `!isUser && message.order_draft`.
- **Durable single-write guard (red-team CRITICAL).** The in-memory `useMutation` guard does NOT survive a page reload — a persisted draft would re-arm the button and double-order. Fix: pass `MessageBubble`/`OrderConfirmCard` an `onPlaced(orderId)` callback that sets `placedOrderId` on that message (keyed by `ts`) via `setMessages`, which re-persists to localStorage. When `message.placedOrderId` is set, the card renders a "✓ Đã đặt hàng #id" link to `/orders/{id}` and never shows the form/button again — surviving reload.

### U3. `OrderConfirmCard` — new `components/chatbot/order-confirm-card.tsx`

Props: `{ draft: OrderDraft; placedOrderId?: number; onPlaced: (id: number) => void }`.

- **Placed state:** if `placedOrderId` is set → render "✓ Đã đặt hàng #id" (Link to `/orders/{id}`), nothing else.
- **Guest gate (inline, not `RequireAuth`):** read `accessToken`/`user`/`isHydrated` from `useAuthStore`. While `!isHydrated` render nothing (avoid login-flash). If hydrated and `!accessToken` → render a `Đăng nhập để đặt hàng` link (`/login?next=<encodeURIComponent(pathname)>`), optionally showing the line list above as a preview.
- **Line list:** render authoritatively from `draft.items` — `product_name`, a prominent `size · color` subline (defense against wrong-variant), `× quantity`, `formatPrice(unit_price)đ`, right-aligned `formatPrice(line_total)đ`; grand total `formatPrice(draft.total)đ`. Thumbnails: reuse the `ProductSourceCards` `useQueries(productsApi.detail)` pattern, but **failure-tolerant** — on error/loading show a placeholder; thumbnails NEVER gate ordering.
- **Warnings:** render every `draft.warnings` entry in a red banner (reuse existing error-banner styling). Lines with `in_stock === false` get an inline "hết hàng" tag and are dimmed.
- **Form (COD only):** `recipientName`/`recipientPhone` prefilled from `useAuthStore().user` (`fullName`/`phone`) with a `useEffect` backfill when `user` hydrates; `recipientAddress` required free-text (no source on `Account`); `note` optional; `recipientEmail` = `user?.email || undefined`. Payment hard-wired `"COD"`.
- **Place order:** `useMutation` → `ordersApi.create({ items: draft.items.map(l => ({ variantId: Number(l.variant_id), quantity: l.quantity })), recipientName, recipientPhone, recipientAddress, recipientEmail, note, paymentMethod: "COD" })`. Bearer auto-attached by `api.post`. On success → `toast("Đặt hàng thành công!", "success")`, `onPlaced(order.id)`, `router.replace('/orders/'+order.id)`. On error → inline banner via `getErrorMessage(err, "Đặt hàng thất bại")`, no navigation.
- **Disable rule:** `disabled = placeOrder.isPending || placeOrder.isSuccess || draft.warnings.length > 0 || draft.items.length === 0 || !recipientName || !recipientPhone || !recipientAddress`.
- Does **not** import `useCartStore` (items come from the draft, not the cart).

---

## Data flow (end to end)

```
User: "mua 2 vợt Astrox 100ZZ size 4U, màu đỏ"
 → RAG /chat: retrieve → get_pricing(product_id) [existing] → model maps 4U/Đỏ → variant_id
 → prepare_order([{product_id, variant_id, quantity:2}])
     · get_pricing (cached) → unit_price (salePrice||price), line_total
     · check_inventory → central row by storeName == "Goodminton HQ - Di An" → in_stock
     · returns canonical order_draft JSON (warnings if any)
 → ChatResponse.order_draft (model prose invites confirm, restates size/color, no total in prose)
 → UI: OrderConfirmCard from order_draft (prefill name/phone, address input, COD)
 → user clicks "Đặt hàng" → ordersApi.create (user JWT) → server re-prices + deducts central stock
 → success → onPlaced(id) marks message (persisted) → "✓ Đã đặt #id", button never re-arms
```

## Error handling & failure-mode mitigations (from red-team)

| Failure | Mitigation (in this design) |
|---|---|
| Duplicate order on reload (persisted draft re-arms button) | `placedOrderId` persisted to the message → card locks to "Đã đặt #id" after success; survives reload |
| 3B emits `quantity:0`/negative | Reject + warning + drop line (never clamp) |
| Hallucinated huge quantity | `quantity > 20` → warning + drop line; `items` capped at 20 |
| Repeat guard cuts flow before `prepare_order` | `MAX_REPEATED_CALLS`→3; forced-final prompt says "chưa tạo được đơn, thử lại", never invents a card |
| 3B maps wrong variant (size/color) | Split size/color rendered prominently + prompt makes model restate chosen size/color |
| Wrong `central_store_name` → all out-of-stock | Grounded default `"Goodminton HQ - Di An"`; **mandatory** end-to-end dry-run build step |
| Partial/empty draft slips through | Any warning OR `items.length === 0` disables ordering; dropped lines always emit a warning |
| Price drift (draft vs charged) | Server is authoritative; `/orders/{id}` shows the real total; card copy is indicative; never send price |
| Stock race at confirm | Server atomic `decrementIfAvailable` → `ORDER_INVENTORY_INSUFFICIENT`; surfaced inline via `getErrorMessage`, no navigation. (Confirm 2302 has a Vietnamese message; else acceptable.) |
| PII in Langfuse | Prompt rule: don't solicit address in chat (card collects it) |
| `/chat` unauth read-amplification | `items[:20]` + `maxItems:20` |

## Scope

**In scope:** buy-intent detection, variant resolution via existing `get_pricing`, `prepare_order` read-only draft, `order_draft` on the chat response, `OrderConfirmCard` with COD checkout reusing `ordersApi.create`, guest→login gate, durable single-write guard.

**Out of scope (YAGNI):** PAYOS/VNPAY in chat; saved-address book (doesn't exist); per-line partial ordering; order editing after placement; multi-store; server-side idempotency keys; Langfuse trace redaction; touching the shared internal-key auth.

## Testing

**RAG** (`tests/test_tools_prepare_order.py`, mirroring `test_tools_similar.py`; `settings.central_store_name` set via fixture):
- schema registered; success draft (prices, total, in_stock); salePrice precedence; null salePrice uses price; out-of-stock → `in_stock=false` + warning; variant-not-found → dropped line + warning + `check_inventory` not called; `quantity<=0` → dropped + warning + `get_pricing` not called; `quantity>20` → dropped + warning; `items` capped at 20.
- `tests/test_chat_router.py`: `_parse_order_draft` (valid → dict; `{"error":...}` → None; non-JSON → None); a `_run_tool_loop` test that a scripted `get_pricing`→`prepare_order` sequence yields `order_draft` on the 3-tuple and the answer invites confirmation.

**UI** (`components/chatbot/__tests__/order-confirm-card.test.tsx`; mock `ordersApi.create`, `productsApi.detail`, `useAuthStore`, `next/navigation`):
- renders lines + total + prefilled name/phone (logged in); guest shows `/login?next=` link and no form; submit calls `ordersApi.create` once with the exact payload (ids `Number()`-coerced, `paymentMethod:"COD"`), then `router.replace('/orders/'+id)` + `onPlaced` fired; second click → no second call; a message with `placedOrderId` renders the "Đã đặt #id" state; any warning (or empty items) disables the button and no call fires; thumbnail-fetch failure still renders + still orderable.

## Build / verification (must-do)

1. **Verify `central_store_name` end to end.** The seeded value `"Goodminton HQ - Di An"` is baked as the default; the plan's final step MUST run a live buy → confirm → placed-order dry run against seeded data to prove stock resolves and an order is created. This is the #1 demo-killer if skipped.
2. RAG: `DATABASE_URL=…goodminton_test uv run pytest` green; `ruff` clean.
3. UI: `tsc` clean + component tests green.

## Open items / assumptions

- `getErrorMessage` mapping for shop-api code `2302` (insufficient stock) to Vietnamese — confirm during UI task; non-blocking if English falls through.
- UI test harness (jest vs vitest + RTL config) — confirm before writing UI test files.
- `toast` import shape (`store/toast-store.ts`) — confirm named export vs hook before copying the checkout call.
