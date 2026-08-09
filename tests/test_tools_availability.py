import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.services.tools import MAX_VARIANTS_PER_LOOKUP, ToolDispatcher


def _variant(variant_id: int, size: str) -> dict:
    return {
        "variantId": variant_id,
        "colorName": "Đỏ",
        "sizeName": size,
        "skuCode": f"AX99-{size}",
        "price": 2890000,
        "salePrice": 2689000,
    }


def _row(store_id: int, name: str, quantity: int, *, central: bool = False) -> dict:
    """One inventory row as shop-api sends it, isCentral included."""
    return {
        "storeId": store_id,
        "storeName": name,
        "isCentral": central,
        "quantity": quantity,
    }


def _dispatcher(pricing: dict, inventory_by_variant: dict[int, list]):
    client = MagicMock()
    client.get_pricing = AsyncMock(return_value=pricing)
    client.check_inventory = AsyncMock(
        side_effect=lambda vid: inventory_by_variant[vid]
    )
    return ToolDispatcher(client, MagicMock()), client


async def _run(dispatcher, product_id=12) -> dict:
    raw = await dispatcher.execute(
        "get_product_availability", {"product_id": product_id}
    )
    return json.loads(raw)


@pytest.mark.asyncio
async def test_returns_price_and_stock_for_every_variant_in_one_call():
    """The whole point of merging the two tools: one call answers both "how much"
    and "is it in stock", so the model never needs a second round trip."""
    dispatcher, client = _dispatcher(
        {
            "productId": 12,
            "productName": "Yonex Astrox 99 Game",
            "variants": [_variant(45, "4U"), _variant(46, "3U")],
        },
        {
            45: [_row(1, "HQ", 7, central=True)],
            46: [_row(1, "HQ", 2, central=True)],
        },
    )

    out = await _run(dispatcher)

    assert out["productId"] == 12
    assert out["productName"] == "Yonex Astrox 99 Game"
    assert [v["variantId"] for v in out["variants"]] == [45, 46]
    # Pricing fields survive untouched alongside the new stock fields.
    assert out["variants"][0]["salePrice"] == 2689000
    assert out["variants"][0]["sizeName"] == "4U"
    assert out["variants"][0]["orderable"] == 7
    assert out["variants"][1]["orderable"] == 2
    assert client.get_pricing.await_count == 1
    assert client.check_inventory.await_count == 2


@pytest.mark.asyncio
async def test_separates_orderable_stock_from_walk_in_branches():
    """An ONLINE order is fulfilled from the central store only, so the two
    numbers must never be added together: reporting a cross-store total as
    available would have the bot promise stock checkout cannot draw on. Branches
    holding nothing are dropped -- noise in an already context-tight prompt."""
    dispatcher, _ = _dispatcher(
        {"productId": 12, "productName": "P", "variants": [_variant(45, "4U")]},
        {
            45: [
                _row(1, "HQ", 3, central=True),
                _row(2, "Q1", 0),
                _row(3, "Q7", 5),
            ]
        },
    )

    variant = (await _run(dispatcher))["variants"][0]

    assert variant["orderable"] == 3  # central only, NOT 3 + 5
    assert [b["storeName"] for b in variant["branches"]] == ["Q7"]
    assert variant["branches"][0]["quantity"] == 5


@pytest.mark.asyncio
async def test_out_of_stock_variant_reports_zero_and_no_stores():
    dispatcher, _ = _dispatcher(
        {"productId": 12, "productName": "P", "variants": [_variant(45, "4U")]},
        {45: [_row(1, "HQ", 0, central=True)]},
    )

    variant = (await _run(dispatcher))["variants"][0]

    assert variant["orderable"] == 0
    assert variant["branches"] == []


@pytest.mark.asyncio
async def test_variant_with_no_inventory_rows_is_out_of_stock():
    """A variant that was never stocked anywhere has no rows at all - it must not
    look the same as "stock unknown"."""
    dispatcher, _ = _dispatcher(
        {"productId": 12, "productName": "P", "variants": [_variant(45, "4U")]},
        {45: []},
    )

    variant = (await _run(dispatcher))["variants"][0]

    assert variant["orderable"] == 0
    assert variant["branches"] == []


@pytest.mark.asyncio
async def test_product_with_no_variants_does_not_call_inventory():
    dispatcher, client = _dispatcher(
        {"productId": 12, "productName": "P", "variants": []}, {}
    )

    out = await _run(dispatcher)

    assert out["variants"] == []
    assert "note" not in out
    client.check_inventory.assert_not_awaited()


@pytest.mark.asyncio
async def test_variant_count_is_capped_and_the_cap_is_reported():
    """The lookup fans out one HTTP call per variant, so it is bounded. A silent
    truncation would read as "these are all the sizes", hence the note."""
    total = MAX_VARIANTS_PER_LOOKUP + 5
    dispatcher, client = _dispatcher(
        {
            "productId": 12,
            "productName": "P",
            "variants": [_variant(100 + i, f"S{i}") for i in range(total)],
        },
        {100 + i: [_row(1, "HQ", 1, central=True)] for i in range(total)},
    )

    out = await _run(dispatcher)

    assert len(out["variants"]) == MAX_VARIANTS_PER_LOOKUP
    assert client.check_inventory.await_count == MAX_VARIANTS_PER_LOOKUP
    assert str(MAX_VARIANTS_PER_LOOKUP) in out["note"] and str(total) in out["note"]


@pytest.mark.asyncio
async def test_inventory_reads_run_concurrently():
    """The merge trades one LLM round trip for N HTTP calls, which only pays off
    if those calls overlap. Deterministic check: with gather every call enters
    before any finishes, whereas sequential awaits would interleave enter/exit."""
    order: list[str] = []

    async def slow_inventory(variant_id: int):
        order.append(f"enter:{variant_id}")
        await asyncio.sleep(0)  # yield to the loop
        order.append(f"exit:{variant_id}")
        return [_row(1, "HQ", 1, central=True)]

    client = MagicMock()
    client.get_pricing = AsyncMock(
        return_value={
            "productId": 12,
            "productName": "P",
            "variants": [_variant(45, "4U"), _variant(46, "3U"), _variant(47, "2U")],
        }
    )
    client.check_inventory = AsyncMock(side_effect=slow_inventory)

    await ToolDispatcher(client, MagicMock()).execute(
        "get_product_availability", {"product_id": 12}
    )

    assert order[:3] == ["enter:45", "enter:46", "enter:47"]


@pytest.mark.asyncio
async def test_unknown_product_returns_the_shared_404_guidance():
    """404 must keep flowing through execute()'s handler, which tells the model to
    stop guessing ids rather than surfacing a stack trace to the customer."""
    client = MagicMock()
    client.get_pricing = AsyncMock(
        side_effect=httpx.HTTPStatusError(
            "not found",
            request=httpx.Request("GET", "http://shop-api/x"),
            response=httpx.Response(404),
        )
    )
    dispatcher = ToolDispatcher(client, MagicMock())

    out = json.loads(
        await dispatcher.execute("get_product_availability", {"product_id": 999})
    )

    assert "error" in out
    assert "Chỉ dùng ID" in out["error"]


@pytest.mark.asyncio
async def test_inventory_failure_fails_the_whole_lookup():
    """Partial stock would be worse than none: the model would confidently report
    "out of stock" for a variant whose read simply failed."""
    client = MagicMock()
    client.get_pricing = AsyncMock(
        return_value={
            "productId": 12,
            "productName": "P",
            "variants": [_variant(45, "4U"), _variant(46, "3U")],
        }
    )
    client.check_inventory = AsyncMock(side_effect=httpx.ConnectError("boom"))
    dispatcher = ToolDispatcher(client, MagicMock())

    out = json.loads(
        await dispatcher.execute("get_product_availability", {"product_id": 12})
    )

    assert out == {"error": "Hệ thống tra cứu tạm thời gặp lỗi."}


@pytest.mark.asyncio
async def test_retired_tool_names_are_rejected():
    """get_pricing and check_inventory are gone; a model that still emits one must
    get a clear unknown-tool error rather than a silent no-op."""
    dispatcher, _ = _dispatcher(
        {"productId": 12, "productName": "P", "variants": []}, {}
    )

    for retired in ("get_pricing", "check_inventory"):
        out = json.loads(await dispatcher.execute(retired, {"product_id": 12}))
        assert out["error"] == f"Unknown tool: {retired}"
