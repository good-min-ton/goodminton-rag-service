# Chatbot Order Flow — Design Spec

> **Scope:** Spec 2 of 3 in the sales-chatbot fix package. Closes **P3** (order flow
> unmanaged — bot keeps suggesting during/after confirmation, loops, forgets the selected
> product) from `docs/superpowers/audits/2026-07-27-chatbot-sales-flow-audit.md`.
> Builds on **Spec 1 (Foundation)** which is already implemented. Spec 3 (tool-result
> sanitize, P4) is separate and out of scope here.

**Date:** 2026-07-27
**Branches:** RAG `feat/chatbot-order-flow` (off `feat/chatbot-foundation`), shop-ui `feat/chatbot-order-flow-ui` (off `feat/chatbot-foundation-ui`). Stacked on the unmerged Foundation — rebase when Foundation merges.
**Repos:** `goodminton-rag-service` (primary), `goodminton-shop-ui` (one small wire change).

---

## 1. Problem (P3, from the audit)

The chatbot has no order/conversation state machine. `order_draft` is derived only from the
current turn and reset every turn; the backend/LLM never learns an order was placed (placement
is frontend-only via `OrderConfirmCard → ordersApi.create`). Consequences: after a product is
selected the bot keeps recommending other categories, recommendations still appear alongside
the confirm card, the conversation loops without closing, and the bot "forgets" the selected
product.

The Foundation (Spec 1) already added server-side `ConversationState` in Redis with **reserved,
unused** `order_status` and `selected_product_id` fields, plus a structured `display_products`
contract and deterministic query understanding (`qu.intent`, `qu.categories`). Spec 2 activates
those reserved fields as an order state machine that gates recommendations.

## 2. State machine

`ConversationState.order_status` becomes an enum with **three behavior-differentiating states**
(the audit's six-state sketch collapses onto these — PRODUCT_SELECTED/CHECKING_STOCK/COMPLETED
do not change pipeline behavior, so they are not modeled; YAGNI):

- **BROWSING** (default / `None`): full Foundation pipeline — category-filtered retrieval + reco + `display_products`.
- **WAITING_CONFIRMATION**: a turn produced an `order_draft` (the confirm card is showing). Suppress NEW-product recommendations (`display_products = []`), keep pricing/stock/prepare_order tools available for changes, and constrain the LLM via a prompt directive to confirm/cancel/change only.
- **ORDER_CONFIRMED**: the order was placed. No suggestions; the bot thanks the customer and offers a fresh search. Exits to BROWSING on a new buy request.

### Transition function (deterministic, server-side)

`next_order_status(current, qu, order_draft_emitted, order_placed_id) -> str` with this
precedence (evaluated top to bottom):

1. **New browse wins:** `qu.categories` is non-empty AND `current != BROWSING` → **BROWSING**
   (the user named a product category → they moved on; reset order context incl. `selected_product_id`).
2. **Draft emitted this turn:** `order_draft_emitted` → **WAITING_CONFIRMATION**
   (a fresh draft is strong evidence of an active order — it outranks a resent stale placement signal).
3. **Placement signal:** `order_placed_id is not None` → **ORDER_CONFIRMED**.
4. Otherwise → keep `current` (default BROWSING).

Rationale for precedence: rule 1 lets a resent stale `order_placed_id` (see §4) never lock the
user out of browsing — a new category always drops back to BROWSING. Rule 2 above rule 3 means
starting a new order after a prior placement re-enters WAITING rather than being pinned to
ORDER_CONFIRMED by the stale signal. A bare `qu.intent == "buy"` with NO category is NOT treated
as a new browse (it is usually a confirm like "mua luôn" / "đồng ý"), so it does not force BROWSING.

Because `order_draft_emitted` is only known after the tool loop, the handler evaluates the
transition twice: once BEFORE the loop with `order_draft_emitted=False` to pick the prompt
directive (this reflects the state carried in from the prior turn — e.g. still WAITING while the
customer changes a size), and once AFTER the loop with the real `order_draft_emitted` to compute
the status persisted + the `display_products` gate.

`selected_product_id` is set from the first item of `order_draft` when a draft is emitted, and
cleared when transitioning to BROWSING via rule 1.

## 3. Behavior gating (in the chat handler)

The handler (Foundation's rewritten `app/routers/chat.py`) computes `order_status` right after
query understanding, then:

- **WAITING_CONFIRMATION or ORDER_CONFIRMED:** force `display_products = []` (no new-product cards) and skip the `cheapest` price-sort reco emphasis. Retrieval STILL runs normally so the tool loop has a valid catalog (e.g. to re-price the selected product on a size change) — only the surfaced cards are suppressed.
- Append a state directive to `system_content`:
  - WAITING_CONFIRMATION: `"Khách đang xem thẻ xác nhận đơn hàng. CHỈ hỗ trợ: xác nhận, hủy, hoặc đổi size/số lượng của đơn hiện tại. TUYỆT ĐỐI KHÔNG gợi ý sản phẩm mới, không liệt kê sản phẩm khác."` — plus, if `selected_product_id`/draft is known, name it so the model doesn't forget it.
  - ORDER_CONFIRMED: `"Đơn hàng của khách đã được đặt thành công. Cảm ơn khách và hỏi khách có cần tìm thêm gì không. KHÔNG gợi ý sản phẩm cho tới khi khách yêu cầu mua món mới."`
- BROWSING: unchanged Foundation behavior.

The updated `order_status` + `selected_product_id` are persisted to Redis as part of the existing
Foundation state-save step.

## 4. Placement signal (wire change)

Decision (recommended): the frontend flags placement on the **next `/chat` turn** — no new
endpoint.

- `ChatRequest` (RAG `app/models/schemas.py` + shop-ui `types.ts`) gains `order_placed_id: int | None = None`.
- shop-ui: the chat panel already stamps `placedOrderId` on a message when an order is placed
  (`markPlaced`). Track the most recent placed order id for the conversation and include it as
  `order_placed_id` on subsequent `sendChat` calls. It may be resent every turn — that is safe
  because transition rule 1 (new category → BROWSING) always overrides a stale placement signal.
- Backend uses it only to drive the transition (rule 2). It never places or mutates an order —
  `prepare_order` stays read-only; the frontend remains the sole order-placer.

If the customer places an order and never sends another message, the backend simply stays in
WAITING_CONFIRMATION — which already suppresses suggestions — so no suggestion leak occurs; the
only difference is the "thank you" ORDER_CONFIRMED tone, which is acceptable to miss.

## 5. Components / files

- RAG `app/services/conversation_state.py`: change `order_status` to a typed literal
  (`"BROWSING" | "WAITING_CONFIRMATION" | "ORDER_CONFIRMED"`, default `"BROWSING"`); keep
  `selected_product_id`.
- RAG new `app/services/order_flow.py`: pure `next_order_status(...)` transition function +
  `ORDER_STATE_DIRECTIVES` prompt strings. Isolated + unit-testable, no I/O.
- RAG `app/routers/chat.py`: call `next_order_status`, gate `display_products`, append the
  directive, set/clear `selected_product_id`, persist. Read `order_placed_id` from the request.
- RAG `app/models/schemas.py`: `ChatRequest.order_placed_id: int | None = None`.
- shop-ui `components/chatbot/types.ts` + `chat-panel.tsx`: add `order_placed_id` to `ChatRequest`
  and send the tracked placed order id.

## 6. Error handling / degradation

- Unknown/absent `order_status` in loaded state → treat as BROWSING.
- All of Spec 1's degradation holds (Redis down → stateless → always BROWSING → full pipeline;
  never 500).
- The state machine is deterministic and never calls the LLM — no new failure surface.

## 7. Testing

RAG (pytest, pure unit — no DB/LLM):
- `next_order_status` precedence: BROWSING + draft → WAITING; WAITING + placed_id → CONFIRMED;
  WAITING + new category → BROWSING (clears selected); CONFIRMED + new category → BROWSING;
  WAITING + no signal → stays WAITING; bare buy-intent (no category) in WAITING does NOT force BROWSING.
- Handler: WAITING/CONFIRMED forces `display_products == []` and appends the right directive;
  BROWSING keeps Foundation behavior; `selected_product_id` set from draft, cleared on new browse;
  `order_placed_id` in request drives CONFIRMED.

shop-ui: `npx tsc --noEmit` exit 0; `order_placed_id` present on the request and populated after placement.

## 8. Non-goals

- Spec 3 (tool-result JSON sanitize, P4).
- No new order-placement path — `prepare_order` read-only, frontend places the order.
- The audit's PRODUCT_SELECTED / CHECKING_STOCK / COMPLETED micro-states (not behavior-bearing).
- Image-search path untouched.

## 9. Open decisions (proceeding with recommended defaults; confirm on review)

- **Placement signal = next-turn flag** (§4), not a new endpoint and not pure inference. Change here
  only affects §4 + the transition rule-2 input.
- **Three effective states** (§2), not the full six. If the TTTN report needs the full six-state
  diagram as labels, we can widen the enum without changing behavior.
