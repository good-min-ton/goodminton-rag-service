import json
from unittest.mock import AsyncMock


from app.models.schemas import ChatResponse, OrderDraft, OrderDraftItem
from app.services.tools import TOOL_SCHEMAS, ToolDispatcher

CENTRAL = "Central Test Store"


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


def _inv(qty, store=CENTRAL, is_central=True):
    """One inventory row. The central store is identified by the isCentral flag
    shop-api sends, not by matching its name against config."""
    return [
        {"storeId": 1, "storeName": store, "isCentral": is_central, "quantity": qty}
    ]


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
    assert line["product_id"] == "12"  # string
    assert line["variant_id"] == "45"  # string
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
    assert line["unit_price"] == 2500000.0  # salePrice wins
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
    """Stock sitting only at a branch is walk-in stock: an ONLINE order is
    fulfilled from the central store, so 50 units elsewhere are still zero here.
    Failing toward out-of-stock also keeps a missing row from reading as
    "unknown, assume fine"."""
    client, dispatcher = _dispatcher(
        _pricing([_variant()]), _inv(50, store="Chi nhanh Q7", is_central=False)
    )
    out = await dispatcher.execute(
        "prepare_order",
        {"items": [{"product_id": 12, "variant_id": 45, "quantity": 1}]},
    )
    draft = json.loads(out)
    assert draft["items"][0]["in_stock"] is False
    assert len(draft["warnings"]) == 1


async def test_prepare_order_ignores_the_store_name():
    """The central store used to be matched by name against config, so renaming
    it in the admin panel silently made every line read out of stock. Identity
    now comes from the flag shop-api sends."""
    client, dispatcher = _dispatcher(
        _pricing([_variant()]),
        _inv(10, store="Ten Kho Vua Doi", is_central=True),
    )
    out = await dispatcher.execute(
        "prepare_order",
        {"items": [{"product_id": 12, "variant_id": 45, "quantity": 2}]},
    )
    draft = json.loads(out)
    assert draft["items"][0]["in_stock"] is True
    assert draft["warnings"] == []


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


async def test_prepare_order_output_round_trips_into_chat_response_model():
    # Pins that the tool's JSON keys/types stay byte-compatible with the
    # ChatResponse/OrderDraft Pydantic contract consumed by the router.
    client, dispatcher = _dispatcher(_pricing([_variant()]), _inv(10))
    out = await dispatcher.execute(
        "prepare_order",
        {"items": [{"product_id": 12, "variant_id": 45, "quantity": 2}]},
    )
    data = json.loads(out)

    resp = ChatResponse(answer="x", sources=[], order_draft=data)

    assert isinstance(resp.order_draft, OrderDraft)
    item = resp.order_draft.items[0]
    assert isinstance(item, OrderDraftItem)
    assert item.product_id == "12"
    assert item.variant_id == "45"
    assert item.product_name == "Vợt Yonex Astrox 100ZZ"
    assert item.size == "4U"
    assert item.color == "Đỏ"
    assert item.quantity == 2
    assert item.unit_price == 3200000.0
    assert item.line_total == 6400000.0
    assert item.in_stock is True
    assert resp.order_draft.total == 6400000.0
    assert resp.order_draft.currency == "VND"
    assert resp.order_draft.warnings == []
