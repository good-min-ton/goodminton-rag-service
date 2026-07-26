# Chatbot Order Assistant (RAG side) Implementation Plan
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Add a read-only `prepare_order` tool that returns a priced, stock-checked `order_draft` on the RAG chat response so the frontend can render a confirm card.
**Architecture:** RAG stays read-only: a new `prepare_order` tool reuses the existing `get_pricing` + `check_inventory` internal GETs, computes a deterministic draft in `ToolDispatcher`, and threads it through `_run_tool_loop` onto a new `ChatResponse.order_draft` Pydantic field. RAG never writes an order and never sees the user JWT; the write is exclusively frontend-side (separate UI plan). shop-api is untouched.
**Tech Stack:** FastAPI, hand-rolled Ollama tool-calling (not LangChain), `httpx` → shop-api `/api/internal/*`, Pydantic v2. `pytest` + `pytest-asyncio` (`asyncio_mode="auto"`), `uv` runner, `ruff`.

## Global Constraints (every task inherits these — do not violate)
- shop-api is NOT modified: no new endpoints, no schema changes. This branch (`feat/chatbot-order-assistant`, forked from `feat/rag-quick-wins`) touches only `goodminton-rag-service`.
- RAG never writes an order. `prepare_order` performs only GET reads via the existing `get_pricing` + `check_inventory`. NO new `ProductClient` method.
- The chat REQUEST is unchanged (`ChatRequest` untouched — no token, no user id). Only the chat RESPONSE gains `order_draft`.
- ONE canonical `order_draft` schema, byte-identical across tool output, Pydantic model, and UI type. Field names/types are load-bearing: `items[]` with `product_id` (str), `variant_id` (str), `product_name` (str), `size` (str|None), `color` (str|None), `quantity` (int), `unit_price` (float), `line_total` (float), `in_stock` (bool); top-level `total` (float), `currency` (str = "VND"), `warnings` (list[str]).
- Payment is COD only (frontend concern; no RAG impact beyond the draft).
- Never silently coerce user intent: invalid `quantity` (`<= 0` or `> 20`) or unknown `variant_id` → append a warning + DROP the line, never clamp and never fabricate a line.
- `unit_price = salePrice if salePrice is not None else price` (null salePrice = no discount, never 0). `line_total = unit_price * quantity`. `total = sum(line_total)` over included lines. All prices are display-only; the server re-prices.
- Central-store availability only: match the inventory row by `storeName == settings.central_store_name`; no match → available `0` (fail toward out-of-stock, never fail-open). `in_stock = central_qty >= quantity`.
- `items` is capped: `maxItems: 20` in the tool schema AND `items[:20]` in code (read-amplification guard).
- Grounded shop-api field names (verified against shop-api DTOs 2026-07-26): pricing = `{productId, productName, variants:[{variantId, colorName, sizeName, skuCode, price, salePrice}]}`; inventory = `[{storeId, storeName, quantity}]`.
- All new tests are pure unit tests using `unittest.mock.AsyncMock` (NO database) — mirror `tests/test_tools_similar.py`. They run with a bare `uv run pytest <file>`; the `DATABASE_URL=…goodminton_test uv run pytest` form is only for DB-backed tests (none here).

---

### Task 1: Canonical `order_draft` contract (Pydantic models)
**Files:**
- Modify: `app/models/schemas.py:17-27` (insert `OrderDraftItem` + `OrderDraft` between `SourceRef` and `ChatResponse`; add `order_draft` field to `ChatResponse`)
- Test: `tests/test_schemas_order_draft.py` (create)
**Interfaces:**
- Consumes: nothing (contract-first).
- Produces: `app.models.schemas.OrderDraftItem`, `app.models.schemas.OrderDraft`, and `ChatResponse.order_draft: OrderDraft | None = None`. Tasks 2 and 3 depend on these exact names/types.

> Placement note: the models are defined BEFORE `ChatResponse` (not "near `SimilarProduct`" as the spec sketch suggested) because `schemas.py` has no `from __future__ import annotations`, so `order_draft: OrderDraft | None` is evaluated eagerly at class-definition time and `OrderDraft` must already exist. `Field` is already imported (line 3); no new imports.

- [ ] **Step 1: Write the failing test**   Create `tests/test_schemas_order_draft.py`:
  ```python
  from app.models.schemas import ChatResponse, OrderDraft, OrderDraftItem


  def test_order_draft_item_fields_and_types():
      item = OrderDraftItem(
          product_id="12",
          variant_id="45",
          product_name="Vợt Yonex Astrox 100ZZ",
          size="4U",
          color="Đỏ",
          quantity=2,
          unit_price=3200000.0,
          line_total=6400000.0,
          in_stock=True,
      )
      assert item.product_id == "12"      # string id convention
      assert item.variant_id == "45"      # string id convention
      assert item.size == "4U"
      assert item.color == "Đỏ"


  def test_order_draft_item_size_color_optional():
      item = OrderDraftItem(
          product_id="1",
          variant_id="2",
          product_name="X",
          quantity=1,
          unit_price=100.0,
          line_total=100.0,
          in_stock=False,
      )
      assert item.size is None
      assert item.color is None


  def test_order_draft_defaults():
      draft = OrderDraft()
      assert draft.items == []
      assert draft.total == 0.0
      assert draft.currency == "VND"
      assert draft.warnings == []


  def test_chat_response_order_draft_defaults_none():
      resp = ChatResponse(answer="hi", sources=[])
      assert resp.order_draft is None


  def test_chat_response_coerces_dict_into_order_draft():
      resp = ChatResponse(
          answer="hi",
          sources=[],
          order_draft={
              "items": [
                  {
                      "product_id": "12",
                      "variant_id": "45",
                      "product_name": "Vợt Yonex Astrox 100ZZ",
                      "size": "4U",
                      "color": "Đỏ",
                      "quantity": 2,
                      "unit_price": 3200000.0,
                      "line_total": 6400000.0,
                      "in_stock": True,
                  }
              ],
              "total": 6400000.0,
              "currency": "VND",
              "warnings": [],
          },
      )
      assert isinstance(resp.order_draft, OrderDraft)
      assert resp.order_draft.items[0].variant_id == "45"
      assert resp.order_draft.total == 6400000.0
  ```
- [ ] **Step 2: Run test to verify it fails**   Run: `uv run pytest tests/test_schemas_order_draft.py -q`   Expected: FAIL with `ImportError: cannot import name 'OrderDraft' from 'app.models.schemas'`.
- [ ] **Step 3: Write minimal implementation**   In `app/models/schemas.py`, insert the two models immediately after the `SourceRef` class (line 19) and before `class ChatResponse` (line 22):
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
  Then add the field to `ChatResponse` (after the existing `products` field, current line 27):
  ```python
      order_draft: OrderDraft | None = None
  ```
- [ ] **Step 4: Run test to verify it passes**   Run: `uv run pytest tests/test_schemas_order_draft.py -q`   Expected: PASS (5 passed).
- [ ] **Step 5: Commit**   Run:
  ```bash
  git add app/models/schemas.py tests/test_schemas_order_draft.py
  git commit -m "feat(schemas): add OrderDraft contract + ChatResponse.order_draft

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 2: `prepare_order` tool + central-store config
**Files:**
- Modify: `app/core/config.py:59-61` (add `central_store_name` after `internal_api_key`)
- Modify: `app/services/tools.py:57-77` (new `TOOL_SCHEMAS` entry after `recommend_similar_products`), `app/services/tools.py:92-96` (dispatch branch after `check_inventory`), `app/services/tools.py:153` (append `_prepare_order` method at end of `ToolDispatcher`)
- Test: `tests/test_tools_prepare_order.py` (create)
**Interfaces:**
- Consumes: `settings.central_store_name` (added in this task); existing `ProductClient.get_pricing(product_id) -> dict` and `ProductClient.check_inventory(variant_id) -> list[dict]`.
- Produces: the `prepare_order` tool emitting the canonical `order_draft` JSON (dict with `items/total/currency/warnings`). Task 3 parses this JSON.

- [ ] **Step 1: Write the failing test**   Create `tests/test_tools_prepare_order.py`:
  ```python
  import json
  from unittest.mock import AsyncMock

  import pytest

  from app.core.config import settings
  from app.services.tools import TOOL_SCHEMAS, ToolDispatcher

  CENTRAL = "Central Test Store"


  @pytest.fixture(autouse=True)
  def _central_store(monkeypatch):
      monkeypatch.setattr(settings, "central_store_name", CENTRAL)


  def _pricing(variants, product_id=12, product_name="Vợt Yonex Astrox 100ZZ"):
      return {"productId": product_id, "productName": product_name, "variants": variants}


  def _variant(variant_id=45, color="Đỏ", size="4U", price=3200000.0, sale=None):
      return {
          "variantId": variant_id,
          "colorName": color,
          "sizeName": size,
          "skuCode": "SKU-1",
          "price": price,
          "salePrice": sale,
      }


  def _inv(qty, store=CENTRAL):
      return [{"storeId": 1, "storeName": store, "quantity": qty}]


  def _dispatcher(pricing_return, inventory_return):
      client = AsyncMock()
      client.get_pricing.return_value = pricing_return
      client.check_inventory.return_value = inventory_return
      return client, ToolDispatcher(product_client=client, similar=AsyncMock())


  def test_prepare_order_schema_present():
      names = {t["function"]["name"] for t in TOOL_SCHEMAS}
      assert "prepare_order" in names


  def test_prepare_order_schema_items_max_20():
      schema = next(t for t in TOOL_SCHEMAS if t["function"]["name"] == "prepare_order")
      assert schema["function"]["parameters"]["properties"]["items"]["maxItems"] == 20


  async def test_prepare_order_success_draft_prices_total_in_stock():
      client, dispatcher = _dispatcher(_pricing([_variant()]), _inv(10))
      out = await dispatcher.execute(
          "prepare_order",
          {"items": [{"product_id": 12, "variant_id": 45, "quantity": 2}]},
      )
      draft = json.loads(out)
      assert draft["currency"] == "VND"
      assert draft["warnings"] == []
      assert len(draft["items"]) == 1
      line = draft["items"][0]
      assert line["product_id"] == "12"      # string
      assert line["variant_id"] == "45"      # string
      assert line["product_name"] == "Vợt Yonex Astrox 100ZZ"
      assert line["size"] == "4U"
      assert line["color"] == "Đỏ"
      assert line["quantity"] == 2
      assert line["unit_price"] == 3200000.0
      assert line["line_total"] == 6400000.0
      assert line["in_stock"] is True
      assert draft["total"] == 6400000.0


  async def test_prepare_order_saleprice_precedence():
      client, dispatcher = _dispatcher(
          _pricing([_variant(price=3200000.0, sale=2500000.0)]), _inv(10)
      )
      out = await dispatcher.execute(
          "prepare_order",
          {"items": [{"product_id": 12, "variant_id": 45, "quantity": 1}]},
      )
      line = json.loads(out)["items"][0]
      assert line["unit_price"] == 2500000.0   # salePrice wins
      assert line["line_total"] == 2500000.0


  async def test_prepare_order_null_saleprice_uses_price():
      client, dispatcher = _dispatcher(
          _pricing([_variant(price=3200000.0, sale=None)]), _inv(10)
      )
      out = await dispatcher.execute(
          "prepare_order",
          {"items": [{"product_id": 12, "variant_id": 45, "quantity": 1}]},
      )
      line = json.loads(out)["items"][0]
      assert line["unit_price"] == 3200000.0


  async def test_prepare_order_out_of_stock_warns_keeps_line():
      client, dispatcher = _dispatcher(_pricing([_variant()]), _inv(1))
      out = await dispatcher.execute(
          "prepare_order",
          {"items": [{"product_id": 12, "variant_id": 45, "quantity": 2}]},
      )
      draft = json.loads(out)
      assert draft["items"][0]["in_stock"] is False
      assert len(draft["warnings"]) == 1
      assert "chỉ còn 1" in draft["warnings"][0]


  async def test_prepare_order_no_central_row_treats_as_zero():
      client, dispatcher = _dispatcher(
          _pricing([_variant()]), _inv(50, store="Some Other Store")
      )
      out = await dispatcher.execute(
          "prepare_order",
          {"items": [{"product_id": 12, "variant_id": 45, "quantity": 1}]},
      )
      draft = json.loads(out)
      assert draft["items"][0]["in_stock"] is False
      assert len(draft["warnings"]) == 1


  async def test_prepare_order_variant_not_found_drops_line_no_inventory_call():
      client, dispatcher = _dispatcher(_pricing([_variant(variant_id=45)]), _inv(10))
      out = await dispatcher.execute(
          "prepare_order",
          {"items": [{"product_id": 12, "variant_id": 999, "quantity": 1}]},
      )
      draft = json.loads(out)
      assert draft["items"] == []
      assert len(draft["warnings"]) == 1
      assert "999" in draft["warnings"][0]
      client.check_inventory.assert_not_called()


  async def test_prepare_order_quantity_zero_drops_line_no_pricing_call():
      client, dispatcher = _dispatcher(_pricing([_variant()]), _inv(10))
      out = await dispatcher.execute(
          "prepare_order",
          {"items": [{"product_id": 12, "variant_id": 45, "quantity": 0}]},
      )
      draft = json.loads(out)
      assert draft["items"] == []
      assert len(draft["warnings"]) == 1
      client.get_pricing.assert_not_called()


  async def test_prepare_order_quantity_over_20_drops_line():
      client, dispatcher = _dispatcher(_pricing([_variant()]), _inv(100))
      out = await dispatcher.execute(
          "prepare_order",
          {"items": [{"product_id": 12, "variant_id": 45, "quantity": 21}]},
      )
      draft = json.loads(out)
      assert draft["items"] == []
      assert len(draft["warnings"]) == 1
      client.get_pricing.assert_not_called()


  async def test_prepare_order_caps_items_at_20():
      client, dispatcher = _dispatcher(_pricing([_variant()]), _inv(100))
      items = [{"product_id": 12, "variant_id": 45, "quantity": 1} for _ in range(25)]
      out = await dispatcher.execute("prepare_order", {"items": items})
      draft = json.loads(out)
      assert len(draft["items"]) == 20
  ```
- [ ] **Step 2: Run test to verify it fails**   Run: `uv run pytest tests/test_tools_prepare_order.py -q`   Expected: FAIL — `test_prepare_order_schema_present` fails (`prepare_order` not in `TOOL_SCHEMAS`) and the dispatch tests return `{"error": "Unknown tool: prepare_order"}` so `json.loads(out)` has no `items` key (KeyError).
- [ ] **Step 3: Write minimal implementation**
  1. In `app/core/config.py`, after `internal_api_key: str | None = None` (line 61):
     ```python
         # Central store whose inventory row prepare_order reads (env: CENTRAL_STORE_NAME)
         central_store_name: str = "Goodminton HQ - Di An"  # Store.name where is_central=true (verified 2026-07-26)
     ```
  2. In `app/services/tools.py`, add a new entry to `TOOL_SCHEMAS` right after the `recommend_similar_products` entry (after its closing `},` on line 76, before the list-closing `]` on line 77):
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
  3. In `ToolDispatcher.execute`, add a dispatch branch immediately after the `check_inventory` branch (after line 95, before the `recommend_similar_products` branch). It sets `result` and falls through to the shared `return json.dumps(result, ensure_ascii=False)`:
     ```python
             elif name == "prepare_order":
                 result = await self._prepare_order(arguments.get("items") or [])
     ```
  4. Append the `_prepare_order` method to `ToolDispatcher` (after `execute`, at end of class, current line 153):
     ```python
         async def _prepare_order(self, items: list[dict]) -> dict:
             """Build a priced, stock-checked order draft. Read-only: reuses
             get_pricing + check_inventory. Rejects (never clamps) invalid
             quantities and drops unknown variants with a warning. Bad int casts
             propagate to execute()'s shared (KeyError, TypeError, ValueError)
             handler."""
             pricing_cache: dict[int, dict] = {}
             lines: list[dict] = []
             warnings: list[str] = []
             total = 0.0

             for raw in items[:20]:  # read-amplification cap
                 product_id = int(raw["product_id"])
                 variant_id = int(raw["variant_id"])
                 quantity = int(raw["quantity"])

                 # Reject, never clamp: invalid quantity -> warn + drop (before any read).
                 if quantity <= 0 or quantity > 20:
                     warnings.append(
                         f"Số lượng {quantity} không hợp lệ cho variant {variant_id}, đã bỏ qua."
                     )
                     continue

                 if product_id not in pricing_cache:
                     pricing_cache[product_id] = await self._client.get_pricing(product_id)
                 pricing = pricing_cache[product_id]

                 variant = next(
                     (
                         v
                         for v in pricing.get("variants", [])
                         if v.get("variantId") == variant_id
                     ),
                     None,
                 )
                 if variant is None:
                     warnings.append(
                         f"variant_id {variant_id} không thuộc sản phẩm {product_id}, đã bỏ qua."
                     )
                     continue

                 sale = variant.get("salePrice")
                 unit_price = float(sale if sale is not None else variant.get("price"))
                 line_total = unit_price * quantity
                 product_name = pricing.get("productName") or ""
                 color = variant.get("colorName")
                 size = variant.get("sizeName")

                 inventory = await self._client.check_inventory(variant_id)
                 central_qty = next(
                     (
                         row.get("quantity", 0)
                         for row in inventory
                         if row.get("storeName") == settings.central_store_name
                     ),
                     0,  # no central row -> fail toward out-of-stock, never fail-open
                 )
                 in_stock = central_qty >= quantity
                 if not in_stock:
                     label = " ".join(x for x in (color, size) if x)
                     warnings.append(
                         f"{product_name} ({label}) chỉ còn {central_qty} tại kho, cần {quantity}."
                     )

                 total += line_total
                 lines.append(
                     {
                         "product_id": str(product_id),
                         "variant_id": str(variant_id),
                         "product_name": product_name,
                         "size": size,
                         "color": color,
                         "quantity": quantity,
                         "unit_price": unit_price,
                         "line_total": line_total,
                         "in_stock": in_stock,
                     }
                 )

             return {
                 "items": lines,
                 "total": total,
                 "currency": "VND",
                 "warnings": warnings,
             }
     ```
- [ ] **Step 4: Run test to verify it passes**   Run: `uv run pytest tests/test_tools_prepare_order.py -q`   Expected: PASS (11 passed).
- [ ] **Step 5: Commit**   Run:
  ```bash
  git add app/core/config.py app/services/tools.py tests/test_tools_prepare_order.py
  git commit -m "feat(tools): add read-only prepare_order tool + central_store_name config

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 3: Thread `order_draft` through the chat handler + harden the repeat guard
**Files:**
- Modify: `app/routers/chat.py:23` (`MAX_REPEATED_CALLS` 2→3), `:83-93` (unpack 3-tuple + pass into `ChatResponse`), `:96-98` (return type), `:109` (local), `:129` (no-tool return), `:150-151` (collector branch), `:156-174` (reword forced-final message + return), `:177-180` (max-iter return), `:247` (append `_parse_order_draft` helper)
- Test: `tests/test_chat_router.py` (create)
**Interfaces:**
- Consumes: the `prepare_order` JSON contract from Task 2; `ChatResponse.order_draft` from Task 1.
- Produces: `_run_tool_loop(...) -> tuple[str, list[dict], dict | None]` and `_parse_order_draft(result: str) -> dict | None`.

- [ ] **Step 1: Write the failing test**   Create `tests/test_chat_router.py`:
  ```python
  import json
  from unittest.mock import AsyncMock

  from app.routers.chat import _parse_order_draft, _run_tool_loop


  def test_parse_order_draft_valid_dict():
      payload = json.dumps({"items": [], "total": 0.0, "currency": "VND", "warnings": []})
      assert _parse_order_draft(payload) == {
          "items": [],
          "total": 0.0,
          "currency": "VND",
          "warnings": [],
      }


  def test_parse_order_draft_error_payload_returns_none():
      assert _parse_order_draft(json.dumps({"error": "boom"})) is None


  def test_parse_order_draft_missing_items_returns_none():
      assert _parse_order_draft(json.dumps({"total": 0.0})) is None


  def test_parse_order_draft_non_json_returns_none():
      assert _parse_order_draft("not json") is None


  class _Dispatcher:
      """Minimal stub: returns a canned tool result per tool name."""

      def __init__(self, results: dict[str, str]) -> None:
          self._results = results
          self.calls: list[str] = []

      async def execute(self, name: str, arguments: dict) -> str:
          self.calls.append(name)
          return self._results[name]


  async def test_run_tool_loop_pricing_then_prepare_order_yields_draft():
      # LLM: turn 1 -> get_pricing, turn 2 -> prepare_order, turn 3 -> text answer.
      llm = AsyncMock()
      llm.chat_with_tools.side_effect = [
          {
              "role": "assistant",
              "content": "",
              "tool_calls": [
                  {"function": {"name": "get_pricing", "arguments": {"product_id": 12}}}
              ],
          },
          {
              "role": "assistant",
              "content": "",
              "tool_calls": [
                  {
                      "function": {
                          "name": "prepare_order",
                          "arguments": {
                              "items": [
                                  {"product_id": 12, "variant_id": 45, "quantity": 2}
                              ]
                          },
                      }
                  }
              ],
          },
          {
              "role": "assistant",
              "content": "Bạn chọn size 4U màu Đỏ, mời bấm XÁC NHẬN trên thẻ đơn hàng.",
              "tool_calls": [],
          },
      ]

      draft_json = json.dumps(
          {
              "items": [
                  {
                      "product_id": "12",
                      "variant_id": "45",
                      "product_name": "Vợt Yonex Astrox 100ZZ",
                      "size": "4U",
                      "color": "Đỏ",
                      "quantity": 2,
                      "unit_price": 3200000.0,
                      "line_total": 6400000.0,
                      "in_stock": True,
                  }
              ],
              "total": 6400000.0,
              "currency": "VND",
              "warnings": [],
          }
      )
      dispatcher = _Dispatcher(
          {
              "get_pricing": json.dumps(
                  {"productId": 12, "productName": "X", "variants": []}
              ),
              "prepare_order": draft_json,
          }
      )

      answer, tool_products, order_draft = await _run_tool_loop(llm, dispatcher, [])

      assert order_draft is not None
      assert order_draft["items"][0]["variant_id"] == "45"
      assert order_draft["total"] == 6400000.0
      assert "XÁC NHẬN" in answer
      assert dispatcher.calls == ["get_pricing", "prepare_order"]


  async def test_run_tool_loop_repeated_calls_force_final_without_draft():
      # Model gets stuck repeating the SAME get_pricing call. After
      # MAX_REPEATED_CALLS (=3) cache-hits the loop forces a tool-free final
      # answer. No prepare_order ran, so order_draft stays None AND the
      # forced-final answer must not claim a confirmation card exists.
      stuck = {
          "role": "assistant",
          "content": "",
          "tool_calls": [
              {"function": {"name": "get_pricing", "arguments": {"product_id": 12}}}
          ],
      }
      llm = AsyncMock()
      # 4 identical tool turns: 1 fresh + 3 cache-hits -> repeats reaches 3.
      llm.chat_with_tools.side_effect = [stuck, stuck, stuck, stuck]
      llm.chat.return_value = "Xin lỗi, mình chưa tạo được đơn, bạn thử lại giúp nhé."

      dispatcher = _Dispatcher(
          {"get_pricing": json.dumps({"productId": 12, "productName": "X", "variants": []})}
      )

      answer, tool_products, order_draft = await _run_tool_loop(llm, dispatcher, [])

      assert order_draft is None          # no prepare_order -> no draft
      assert llm.chat.await_count == 1    # forced-final path was taken
      assert answer == "Xin lỗi, mình chưa tạo được đơn, bạn thử lại giúp nhé."
      assert dispatcher.calls == ["get_pricing"]  # repeats are cache hits, not re-executed
  ```
- [ ] **Step 2: Run test to verify it fails**   Run: `uv run pytest tests/test_chat_router.py -q`   Expected: FAIL — `ImportError: cannot import name '_parse_order_draft' from 'app.routers.chat'`.
- [ ] **Step 3: Write minimal implementation**   Apply these edits to `app/routers/chat.py`:
  1. Line 23: `MAX_REPEATED_CALLS = 2` → `MAX_REPEATED_CALLS = 3`.
  2. Lines 83-85 (handler call): change
     ```python
             answer, tool_products = await _run_tool_loop(
                 llm_svc, tool_dispatcher, messages
             )
     ```
     to
     ```python
             answer, tool_products, order_draft = await _run_tool_loop(
                 llm_svc, tool_dispatcher, messages
             )
     ```
  3. Lines 89-93 (`ChatResponse`): add the `order_draft` kwarg:
     ```python
             return ChatResponse(
                 answer=answer,
                 sources=_unique_sources(chunks),
                 products=recommended,
                 order_draft=order_draft,
             )
     ```
  4. Lines 96-98 (signature return type): `-> tuple[str, list[dict]]:` → `-> tuple[str, list[dict], dict | None]:`.
  5. After line 109 (`tool_products: list[dict] = []`), add:
     ```python
         order_draft: dict | None = None
     ```
  6. Line 129 (no-tool return): `return msg.get("content") or "", tool_products` → `return msg.get("content") or "", tool_products, order_draft`.
  7. Lines 150-151 (collector): extend with the `prepare_order` branch:
     ```python
                     if name == "recommend_similar_products":
                         _collect_tool_products(result, tool_products)
                     elif name == "prepare_order":
                         parsed = _parse_order_draft(result)
                         if parsed is not None:
                             order_draft = parsed  # last successful prepare_order wins
     ```
  8. Lines 156-164 (forced-final system message): reword so it never invents a card when no draft exists:
     ```python
                 messages.append(
                     {
                         "role": "system",
                         "content": (
                             "Dừng gọi công cụ. Trả lời khách bằng thông tin hiện có. "
                             "Nếu bạn CHƯA tạo được đơn hàng nháp, nói rõ là chưa tạo "
                             "được đơn và mời khách thử lại — TUYỆT ĐỐI KHÔNG nói đã "
                             "tạo thẻ xác nhận hay đã đặt đơn. Nếu thiếu dữ liệu khác, "
                             "nói rõ là không có thông tin và mời khách liên hệ shop."
                         ),
                     }
                 )
     ```
  9. Line 174 (forced-final return): `return final, tool_products` → `return final, tool_products, order_draft`.
  10. Lines 177-180 (max-iterations return): add `order_draft` as the third element:
      ```python
          return (
              "Xin lỗi, mình không xử lý được yêu cầu này. Vui lòng liên hệ shop.",
              tool_products,
              order_draft,
          )
      ```
  11. After `_collect_tool_products` (after line 247), add the helper:
      ```python
      def _parse_order_draft(result: str) -> dict | None:
          """Parse a prepare_order tool result into an order_draft dict, or None.

          Centralized {"error": ...} payloads and non-JSON strings yield no draft
          (present-or-absent, not inferred from prose like _extract_recommended).
          """
          try:
              data = json.loads(result)
          except (ValueError, TypeError):
              return None
          if not isinstance(data, dict) or "items" not in data or "error" in data:
              return None
          return data
      ```
- [ ] **Step 4: Run test to verify it passes**   Run: `uv run pytest tests/test_chat_router.py -q`   Expected: PASS (6 passed). (Langfuse instrumentation is non-blocking — spans are buffered client-side — so `_run_tool_loop` runs without any network call.)
- [ ] **Step 5: Commit**   Run:
  ```bash
  git add app/routers/chat.py tests/test_chat_router.py
  git commit -m "feat(chat): thread order_draft through tool loop + harden repeat guard

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 4: Ordering rules in the system prompt
**Files:**
- Modify: `app/core/prompts.py:4-8` (extend the "QUY TẮC BẮT BUỘC VỀ GIÁ VÀ TỒN KHO" block with ordering rules)
- Test: `tests/test_prompts_ordering.py` (create)
**Interfaces:**
- Consumes: nothing (prose only; reuses the `product_id` allow-list already injected in `chat.py`).
- Produces: no code symbols. The new prose introduces NO `{`/`}` braces so `SYSTEM_PROMPT.format(context=...)` in `chat.py` stays intact.

- [ ] **Step 1: Write the failing test**   Create `tests/test_prompts_ordering.py`:
  ```python
  from app.core.prompts import SYSTEM_PROMPT


  def test_system_prompt_has_ordering_rules():
      for keyword in (
          "get_pricing(product_id)",
          "prepare_order(items)",
          "BẤM XÁC NHẬN",
          "TUYỆT ĐỐI không bịa product_id/variant_id",
          "KHÔNG nêu lại tổng tiền bằng chữ",
          "KHÔNG nói đơn đã được đặt",
          "KHÔNG hỏi địa chỉ giao hàng trong chat",
      ):
          assert keyword in SYSTEM_PROMPT, keyword


  def test_system_prompt_still_formats_with_context():
      # No stray braces were introduced; .format still works.
      assert "seeded-context-marker" in SYSTEM_PROMPT.format(
          context="seeded-context-marker"
      )
  ```
- [ ] **Step 2: Run test to verify it fails**   Run: `uv run pytest tests/test_prompts_ordering.py -q`   Expected: FAIL — `test_system_prompt_has_ordering_rules` raises `AssertionError: prepare_order(items)` (the ordering block is not yet in the prompt).
- [ ] **Step 3: Write minimal implementation**   In `app/core/prompts.py`, insert the ordering block into the "QUY TẮC BẮT BUỘC VỀ GIÁ VÀ TỒN KHO" section — after the current line 8 bullet (`- Nếu user hỏi "còn hàng không" ...`) and before the blank line preceding `Quy tắc tư vấn:` (line 10). The block is plain text inside the existing triple-quoted `SYSTEM_PROMPT`; keep the exact wording (no `{`/`}`):
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
- [ ] **Step 4: Run test to verify it passes**   Run: `uv run pytest tests/test_prompts_ordering.py -q`   Expected: PASS (2 passed).
- [ ] **Step 5: Commit**   Run:
  ```bash
  git add app/core/prompts.py tests/test_prompts_ordering.py
  git commit -m "feat(prompts): add order-flow rules to system prompt

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

## Final verification (after all four tasks)
- Full RAG suite green: `uv run pytest -q` (the new files run without `DATABASE_URL`; DB-backed tests skip cleanly when it is unset — that is expected).
- Lint clean: `uv run ruff check app tests`.
- **Central-store stock resolves (RAG-side — executable on THIS branch, the #1 demo-killer if skipped):** against seeded data, confirm `settings.central_store_name` (`"Goodminton HQ - Di An"`, override via `CENTRAL_STORE_NAME`) matches a real `check_inventory` row so a `prepare_order` on an in-stock variant returns `in_stock=true` with no spurious out-of-stock warning. Quick check: start the RAG service and `POST /chat` a buy request (or drive `ToolDispatcher.execute("prepare_order", …)` directly) and inspect the returned `order_draft`. If the name is wrong, every line reads available=0 and spuriously warns out-of-stock (fail-safe, not fail-open).
- **Placed-order round trip is UI-gated (do NOT block RAG completion on it):** the actual buy → confirm → placed-order flow is executed and closed by the UI plan via `ordersApi.create` under the user's JWT. This RAG branch is read-only by design and cannot place an order — the RAG plan is complete once the tool/draft/threading tasks are green and central-store stock resolves above.
