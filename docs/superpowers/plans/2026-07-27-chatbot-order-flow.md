# Chatbot Order Flow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic order state machine so the chatbot stops recommending during/after order confirmation and remembers the order in progress — closing audit problem P3.

**Architecture:** A pure, I/O-free state machine (`order_flow.py`) computes `order_status` (BROWSING / WAITING_CONFIRMATION / ORDER_CONFIRMED) from the loaded conversation state, the query-understanding result, whether this turn emitted an `order_draft`, and a frontend `order_placed_id` signal. The chat handler wires it: it picks a prompt directive from the incoming state, and after the tool loop it computes the final state, blanks `display_products` while an order is pending/placed, remembers `selected_product_id`, and persists the status. `prepare_order` stays read-only; the frontend still places the order.

**Tech Stack:** FastAPI, Python (`uv`); Next.js 16 + TypeScript (shop-ui). Builds on Spec 1 (Foundation).

**Spec:** `docs/superpowers/specs/2026-07-27-chatbot-order-flow-design.md`

## Global Constraints

- Branches: RAG `feat/chatbot-order-flow` (already created off `feat/chatbot-foundation`), shop-ui `feat/chatbot-order-flow-ui` (already created off `feat/chatbot-foundation-ui`). Both already checked out.
- The state machine is DETERMINISTIC — no LLM call, no I/O — and lives in pure functions so it is fully unit-testable.
- Transition precedence (exact): (1) `qu.categories` non-empty AND current != BROWSING → BROWSING (reset, clear `selected_product_id`); (2) `order_draft_emitted` → WAITING_CONFIRMATION; (3) `order_placed_id is not None` → ORDER_CONFIRMED; (4) keep current (default BROWSING).
- While `order_status` ∈ {WAITING_CONFIRMATION, ORDER_CONFIRMED}: `display_products` MUST be `[]` (no new-product cards) and the price-sort reco step is skipped; retrieval STILL runs (tools need a valid catalog). A state directive is appended to the system prompt.
- `prepare_order` read-only; frontend is the sole order-placer. `order_placed_id` only drives the transition — the backend never places/mutates an order.
- Additive wire change: `ChatRequest.order_placed_id: int | None = None`. `ConversationState.order_status` already exists (Foundation) — retype to the literal with default `"BROWSING"`; `conversation_state` in `ChatResponse` already carries it, so no `ChatResponse` change.
- All Foundation degradation holds (Redis down → BROWSING → full pipeline; never 500).
- RAG tests: `uv run pytest`; DB tests need `DATABASE_URL=postgresql://admin:postgresql123@localhost:5433/goodminton_test`. Lint gate: run `uv run ruff format .` then confirm `uv run ruff check .` AND `uv run ruff format --check .` clean (whole repo) before every commit; no unused imports (F401).
- shop-ui: no test runner — gate is `npx tsc --noEmit` exit 0.
- Non-goals: Spec 3 (tool-result sanitize, P4); image-search path; the audit's non-behavioral micro-states (PRODUCT_SELECTED/CHECKING_STOCK/COMPLETED).

---

### Task 1: Order-flow state machine (pure logic) + typed `order_status`

**Files:**
- Create: `app/services/order_flow.py`
- Modify: `app/services/conversation_state.py` (the `order_status` field)
- Test: `tests/test_order_flow.py`

**Interfaces:**
- Consumes: `QueryUnderstanding` (Foundation, `app/services/query_understanding.py`) — only its `.categories: list[str]` is read.
- Produces:
  - constants `BROWSING`, `WAITING_CONFIRMATION`, `ORDER_CONFIRMED` (str).
  - `next_order_status(current: str | None, qu: QueryUnderstanding, order_draft_emitted: bool, order_placed_id: int | None) -> str`.
  - `suppresses_recommendations(status: str | None) -> bool`.
  - `order_directive(status: str | None) -> str` (text to append to the system prompt; `""` for BROWSING).

- [ ] **Step 1: Write the failing tests**

`tests/test_order_flow.py`:

```python
from app.services.order_flow import (
    BROWSING,
    ORDER_CONFIRMED,
    WAITING_CONFIRMATION,
    next_order_status,
    order_directive,
    suppresses_recommendations,
)
from app.services.query_understanding import QueryUnderstanding


def _qu(categories=None):
    return QueryUnderstanding(categories=categories or [], retrieval_query="x")


def test_browsing_plus_draft_enters_waiting():
    assert next_order_status(BROWSING, _qu(), order_draft_emitted=True, order_placed_id=None) == WAITING_CONFIRMATION


def test_waiting_plus_placed_id_confirms():
    assert next_order_status(WAITING_CONFIRMATION, _qu(), order_draft_emitted=False, order_placed_id=55) == ORDER_CONFIRMED


def test_waiting_plus_new_category_resets_to_browsing():
    assert next_order_status(WAITING_CONFIRMATION, _qu(["Áo cầu lông"]), order_draft_emitted=False, order_placed_id=None) == BROWSING


def test_confirmed_plus_new_category_resets_to_browsing():
    assert next_order_status(ORDER_CONFIRMED, _qu(["Giày cầu lông"]), order_draft_emitted=False, order_placed_id=None) == BROWSING


def test_new_category_while_already_browsing_stays_browsing():
    # rule 1 only fires when current != BROWSING; a category in BROWSING is normal browsing
    assert next_order_status(BROWSING, _qu(["Quần cầu lông"]), order_draft_emitted=False, order_placed_id=None) == BROWSING


def test_draft_outranks_stale_placed_id():
    # new order started after a prior placement (stale id resent) -> WAITING, not CONFIRMED
    assert next_order_status(ORDER_CONFIRMED, _qu(), order_draft_emitted=True, order_placed_id=99) == WAITING_CONFIRMATION


def test_waiting_no_signal_stays_waiting():
    assert next_order_status(WAITING_CONFIRMATION, _qu(), order_draft_emitted=False, order_placed_id=None) == WAITING_CONFIRMATION


def test_none_current_defaults_browsing():
    assert next_order_status(None, _qu(), order_draft_emitted=False, order_placed_id=None) == BROWSING


def test_suppression_flags():
    assert suppresses_recommendations(WAITING_CONFIRMATION) is True
    assert suppresses_recommendations(ORDER_CONFIRMED) is True
    assert suppresses_recommendations(BROWSING) is False
    assert suppresses_recommendations(None) is False


def test_directive_text():
    assert "xác nhận" in order_directive(WAITING_CONFIRMATION).lower()
    assert order_directive(BROWSING) == ""
    assert "đặt" in order_directive(ORDER_CONFIRMED).lower()
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_order_flow.py -v`
Expected: FAIL (`ModuleNotFoundError: app.services.order_flow`).

- [ ] **Step 3: Implement `order_flow.py`**

```python
"""Deterministic order-flow state machine for the chat endpoint. No I/O, no LLM."""

from app.services.query_understanding import QueryUnderstanding

BROWSING = "BROWSING"
WAITING_CONFIRMATION = "WAITING_CONFIRMATION"
ORDER_CONFIRMED = "ORDER_CONFIRMED"

_SUPPRESS = {WAITING_CONFIRMATION, ORDER_CONFIRMED}

_DIRECTIVES = {
    WAITING_CONFIRMATION: (
        "\n\nKhách đang xem thẻ xác nhận đơn hàng. CHỈ hỗ trợ: xác nhận, hủy, hoặc "
        "đổi size/số lượng của đơn hiện tại. TUYỆT ĐỐI KHÔNG gợi ý sản phẩm mới, "
        "không liệt kê sản phẩm khác."
    ),
    ORDER_CONFIRMED: (
        "\n\nĐơn hàng của khách đã được đặt thành công. Cảm ơn khách và hỏi khách có "
        "cần tìm thêm gì không. KHÔNG gợi ý sản phẩm cho tới khi khách yêu cầu mua "
        "món mới."
    ),
}


def next_order_status(
    current: str | None,
    qu: QueryUnderstanding,
    order_draft_emitted: bool,
    order_placed_id: int | None,
) -> str:
    cur = current or BROWSING
    if qu.categories and cur != BROWSING:  # new browse resets any order context
        return BROWSING
    if order_draft_emitted:  # a fresh draft outranks a stale placement signal
        return WAITING_CONFIRMATION
    if order_placed_id is not None:
        return ORDER_CONFIRMED
    return cur


def suppresses_recommendations(status: str | None) -> bool:
    return status in _SUPPRESS


def order_directive(status: str | None) -> str:
    return _DIRECTIVES.get(status or BROWSING, "")
```

- [ ] **Step 4: Retype `order_status` in `ConversationState`**

In `app/services/conversation_state.py`, change the field (currently `order_status: str | None = None`) to default BROWSING:

```python
    order_status: str = "BROWSING"
```

(Leave `selected_product_id: int | None = None` as-is. Keep the `intent`/`categories`/`price_preference` fields unchanged.)

- [ ] **Step 5: Run to verify pass + no Foundation regression**

Run: `uv run pytest tests/test_order_flow.py tests/test_conversation_state.py -v`
Expected: PASS. If any Foundation `test_conversation_state.py` case asserted `order_status is None` (it should only compare whole `ConversationState()` objects, which still match), update the expectation to the new default.

- [ ] **Step 6: Lint + commit**

Run: `uv run ruff format app/services/order_flow.py app/services/conversation_state.py tests/test_order_flow.py && uv run ruff check . && uv run ruff format --check .`
```bash
git add app/services/order_flow.py app/services/conversation_state.py tests/test_order_flow.py
git commit -m "feat(chat): order-flow state machine (pure) + BROWSING default

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Wire the state machine into the chat handler + `order_placed_id`

**Files:**
- Modify: `app/models/schemas.py` (`ChatRequest`)
- Modify: `app/routers/chat.py` (the `chat` handler, lines 26-119)
- Test: `tests/test_chat_router.py` (extend)

**Interfaces:**
- Consumes: `order_flow.next_order_status`, `suppresses_recommendations`, `order_directive`, the constant `BROWSING`; the existing `_run_tool_loop` (returns `(answer, tool_products, order_draft)`), `_structured_display_products`, `_sort_by_price`.
- Produces: `ChatRequest.order_placed_id: int | None = None`; the handler sets `state.order_status` + `state.selected_product_id` and blanks `display_products` when suppressed.

- [ ] **Step 1: Add `order_placed_id` to `ChatRequest`**

In `app/models/schemas.py`, `ChatRequest` (after `session_id`, line 16):

```python
    # Set by the frontend on the turn(s) after it places an order, so the backend
    # can advance the order state machine to ORDER_CONFIRMED. Backend never places
    # an order — this is a read-only signal.
    order_placed_id: int | None = None
```

- [ ] **Step 2: Write the failing test for the gating helper usage**

Add to `tests/test_chat_router.py` (pure test of the gate composition the handler will use — no DB/LLM):

```python
from app.services.order_flow import (
    BROWSING,
    ORDER_CONFIRMED,
    WAITING_CONFIRMATION,
    suppresses_recommendations,
)


def test_display_blanked_when_order_pending():
    display = [1, 2, 3]
    gated = [] if suppresses_recommendations(WAITING_CONFIRMATION) else display
    assert gated == []
    gated2 = [] if suppresses_recommendations(ORDER_CONFIRMED) else display
    assert gated2 == []
    gated3 = [] if suppresses_recommendations(BROWSING) else display
    assert gated3 == [1, 2, 3]
```

(This locks the gate semantics the handler relies on; the full handler path is covered by the pure `order_flow` tests in Task 1 + the manual E2E below.)

- [ ] **Step 3: Run to verify it fails/passes appropriately**

Run: `uv run pytest tests/test_chat_router.py -k order_pending -v`
Expected: PASS once the import resolves (order_flow exists from Task 1); this test guards against a future regression of the gate meaning. If import fails, Task 1 wasn't completed first.

- [ ] **Step 4: Wire the handler**

In `app/routers/chat.py`:

Add imports at the top (with the other `app.services` imports):

```python
from app.services.order_flow import (
    BROWSING,
    next_order_status,
    order_directive,
    suppresses_recommendations,
)
```

After `qu = await qu_svc.analyze(query, state)` (line 46), compute the incoming status (draft not yet known) for the prompt directive:

```python
        incoming_status = next_order_status(
            state.order_status, qu, order_draft_emitted=False,
            order_placed_id=request.order_placed_id,
        )
```

Append the directive to `system_content` — after the catalog `if/else` block (right before `messages: list[dict] = [...]`, ~line 88):

```python
        system_content += order_directive(incoming_status)
```

After the tool loop (`answer, tool_products, order_draft = await _run_tool_loop(...)`, ~line 95), replace the display + persist block (lines 97-107) with:

```python
        order_draft_emitted = order_draft is not None
        final_status = next_order_status(
            incoming_status, qu, order_draft_emitted, request.order_placed_id
        )

        display = _structured_display_products(
            chunks, tool_products, settings.chat_display_products_max
        )
        if suppresses_recommendations(final_status):
            display = []  # no new-product cards while an order is pending/placed
        elif qu.price_preference == "cheapest" and display:
            display = await _sort_by_price(http_request.app.state.http, display)

        # Remember the product under order; clear it when the user moves on to browsing.
        if order_draft_emitted:
            items = order_draft.get("items") or []
            if items:
                try:
                    state.selected_product_id = int(items[0]["product_id"])
                except (KeyError, ValueError, TypeError):
                    pass
        if final_status == BROWSING:
            state.selected_product_id = None

        # Persist scope + order status for the next turn.
        state.categories = qu.categories or state.categories
        state.intent = qu.intent
        state.price_preference = qu.price_preference
        state.order_status = final_status
        await state_store.save(request.session_id, state)
```

(The `return ChatResponse(...)` block is unchanged — `conversation_state=state` now carries the updated `order_status`/`selected_product_id`.)

- [ ] **Step 5: Run the chat tests + full suite regression**

Run: `uv run pytest tests/test_chat_router.py -v` then `DATABASE_URL=postgresql://admin:postgresql123@localhost:5433/goodminton_test uv run pytest -q`
Expected: all pass (existing tool-loop / structured-display tests + the new gate test).

- [ ] **Step 6: Lint + commit**

Run: `uv run ruff format app/routers/chat.py app/models/schemas.py tests/test_chat_router.py && uv run ruff check . && uv run ruff format --check .`
```bash
git add app/routers/chat.py app/models/schemas.py tests/test_chat_router.py
git commit -m "feat(chat): drive order state machine + suppress recos during confirm/placed

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: shop-ui — send `order_placed_id` after placement

**Files:** (branch `feat/chatbot-order-flow-ui`, already checked out)
- Modify: `components/chatbot/types.ts` (`ChatRequest`)
- Modify: `components/chatbot/chat-panel.tsx` (track placed order id + send it)

**Interfaces:**
- Produces: `ChatRequest.order_placed_id?: number`; the panel sends the most recent placed order id on subsequent `/chat` calls.

- [ ] **Step 1: Add the field to the `ChatRequest` type**

In `components/chatbot/types.ts`, `ChatRequest`:

```ts
export interface ChatRequest {
  message: string;
  chat_history?: { role: ChatRole; content: string }[];
  session_id?: string;
  /** Set after the frontend places an order, so the backend advances its order
   *  state machine to ORDER_CONFIRMED. Read-only signal; backend never places. */
  order_placed_id?: number;
}
```

- [ ] **Step 2: Track + send the placed order id**

In `components/chatbot/chat-panel.tsx`: the panel already has `markPlaced(ts, orderId)` which stamps `placedOrderId` on a message. Derive the most recent placed order id from `messages` at send time and pass it to `sendChat`. In the `send` callback, before the `sendChat({...})` call, compute:

```tsx
        const placedId = [...messages]
          .reverse()
          .find((m) => m.placedOrderId != null)?.placedOrderId;
```

and add `order_placed_id: placedId,` to the `sendChat({ message, chat_history, session_id, ... })` object. (`messages` is already a dependency of the `send` callback.)

- [ ] **Step 3: Type gate**

Run: `cd /home/mgriffe-work/Desktop/TTTN/goodminton-shop-ui && npx tsc --noEmit`
Expected: exit 0.

- [ ] **Step 4: Commit**

```bash
git add components/chatbot/types.ts components/chatbot/chat-panel.tsx
git commit -m "feat(chatbot): send order_placed_id so backend reaches ORDER_CONFIRMED

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Manual end-to-end verification (after all tasks; full stack up)

1. Browse "mua vợt Astrox 99" → get a draft (confirm card). Observe: NO other product cards under that answer (`display_products` blanked); state is WAITING_CONFIRMATION.
2. Type "đổi size 4U" → bot re-prices the SAME product (no new-product suggestions).
3. Click confirm on the card (order placed) → next message → bot thanks / offers a fresh search, no product cards (ORDER_CONFIRMED).
4. Type "cho xem giày" → cards return (new category → BROWSING).
5. Regression: a normal browse with no order still shows cards; Redis stopped → chat still responds (always BROWSING).

---

## Self-Review (against the spec)

1. **Spec coverage:** §2 state machine → Task 1 (`next_order_status`); §3 gating (display=[] + directive) → Task 2; §4 `order_placed_id` wire → Task 2 (RAG) + Task 3 (UI); §5 files all covered; §6 degradation (unknown status → BROWSING via `current or BROWSING`); §7 tests → Task 1 exhaustive pure tests + Task 2 gate test + manual E2E; §8 non-goals honored.
2. **Placeholder scan:** none — every step has concrete code.
3. **Type consistency:** `next_order_status(current, qu, order_draft_emitted, order_placed_id)` signature identical in Task 1 def and Task 2 call; `order_status: str = "BROWSING"` (Task 1) matches handler reads/writes (Task 2); `order_placed_id: int | None` (RAG) ↔ `order_placed_id?: number` (UI); `suppresses_recommendations`/`order_directive` names consistent across tasks.
