"""start_order builds the picker payload the frontend renders.

It exists to take variant selection away from the LLM. Asking "which size?" in
prose cost a generation per question and then required the model to map a
free-text reply back to a variant_id, which is where wrong-variant orders came
from. Here the model only names the product.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.schemas import ChatResponse, OrderSelection
from app.routers.chat import _parse_order_selection
from app.services.tools import TOOL_SCHEMAS, ToolDispatcher


def _variant(variant_id, size, color="Đỏ", price=2890000.0, sale=None):
    return {
        "variantId": variant_id,
        "colorName": color,
        "sizeName": size,
        "skuCode": f"SKU-{variant_id}",
        "price": price,
        "salePrice": sale,
    }


def _row(store_id, name, quantity, *, central=False):
    return {
        "storeId": store_id,
        "storeName": name,
        "isCentral": central,
        "quantity": quantity,
    }


def _dispatcher(variants, inventory_by_variant):
    client = MagicMock()
    client.get_pricing = AsyncMock(
        return_value={
            "productId": 12,
            "productName": "Yonex Astrox 99 Game",
            "variants": variants,
        }
    )
    client.check_inventory = AsyncMock(
        side_effect=lambda vid: inventory_by_variant[vid]
    )
    return ToolDispatcher(client, MagicMock()), client


async def _run(dispatcher, product_id=12) -> dict:
    return json.loads(
        await dispatcher.execute("start_order", {"product_id": product_id})
    )


def test_schema_is_registered():
    assert "start_order" in {t["function"]["name"] for t in TOOL_SCHEMAS}


@pytest.mark.asyncio
async def test_every_variant_becomes_a_priced_option():
    dispatcher, client = _dispatcher(
        [_variant(45, "4U"), _variant(46, "3U")],
        {45: [_row(1, "HQ", 7, central=True)], 46: [_row(1, "HQ", 2, central=True)]},
    )

    out = await _run(dispatcher)

    assert out["product_id"] == "12"
    assert out["product_name"] == "Yonex Astrox 99 Game"
    assert out["currency"] == "VND"
    assert [o["variant_id"] for o in out["options"]] == ["45", "46"]
    assert [o["size"] for o in out["options"]] == ["4U", "3U"]
    assert out["options"][0]["orderable"] == 7
    assert client.get_pricing.await_count == 1


@pytest.mark.asyncio
async def test_unit_price_resolves_sale_price_so_the_ui_never_repeats_the_rule():
    dispatcher, _ = _dispatcher(
        [_variant(45, "4U", price=2890000.0, sale=2689000.0)],
        {45: [_row(1, "HQ", 5, central=True)]},
    )

    option = (await _run(dispatcher))["options"][0]

    assert option["unit_price"] == 2689000.0


@pytest.mark.asyncio
async def test_orderable_counts_only_the_central_store():
    """The picker caps its quantity stepper at `orderable`, so this number must
    be what checkout can actually deduct - never a cross-store total."""
    dispatcher, _ = _dispatcher(
        [_variant(45, "4U")],
        {
            45: [
                _row(1, "HQ", 2, central=True),
                _row(2, "Q7", 9),
                _row(3, "Q1", 0),
            ]
        },
    )

    option = (await _run(dispatcher))["options"][0]

    assert option["orderable"] == 2  # not 11
    assert [b["store_name"] for b in option["branches"]] == ["Q7"]
    assert option["branches"][0]["quantity"] == 9


@pytest.mark.asyncio
async def test_out_of_stock_variants_are_kept_so_the_picker_can_grey_them():
    """Dropping them would tell the customer the size does not exist. Greyed out
    with a branch hint is the honest answer."""
    dispatcher, _ = _dispatcher(
        [_variant(45, "4U"), _variant(46, "3U")],
        {
            45: [_row(1, "HQ", 0, central=True), _row(2, "Q7", 4)],
            46: [_row(1, "HQ", 3, central=True)],
        },
    )

    options = (await _run(dispatcher))["options"]

    assert len(options) == 2
    assert options[0]["orderable"] == 0
    assert options[0]["branches"][0]["store_name"] == "Q7"


@pytest.mark.asyncio
async def test_variant_without_colour_is_still_offered():
    """Vợt and shuttles carry no colour: color_id is nullable, so a picker that
    assumed a colour existed would offer nothing at all."""
    dispatcher, _ = _dispatcher(
        [_variant(45, "4U", color=None)], {45: [_row(1, "HQ", 5, central=True)]}
    )

    option = (await _run(dispatcher))["options"][0]

    assert option["color"] is None
    assert option["size"] == "4U"


@pytest.mark.asyncio
async def test_unknown_product_returns_the_shared_error():
    client = MagicMock()
    client.get_pricing = AsyncMock(side_effect=KeyError("nope"))
    dispatcher = ToolDispatcher(client, MagicMock())

    out = json.loads(await dispatcher.execute("start_order", {"product_id": 999}))

    assert "error" in out


class TestParsing:
    """_parse_order_selection decides whether a picker reaches the customer."""

    def test_valid_payload_passes_through(self):
        payload = json.dumps(
            {
                "product_id": "12",
                "product_name": "P",
                "currency": "VND",
                "options": [
                    {
                        "variant_id": "45",
                        "size": "4U",
                        "color": None,
                        "unit_price": 100.0,
                        "orderable": 2,
                        "branches": [],
                    }
                ],
            }
        )

        assert _parse_order_selection(payload) is not None

    def test_error_payload_yields_nothing(self):
        assert _parse_order_selection(json.dumps({"error": "boom"})) is None

    def test_empty_options_yields_nothing(self):
        """A picker with nothing to pick reads as a broken card; the model's own
        "shop chua co san pham nay" is the better answer."""
        payload = json.dumps({"product_id": "12", "product_name": "P", "options": []})

        assert _parse_order_selection(payload) is None

    def test_non_json_yields_nothing(self):
        assert _parse_order_selection("mời bạn chọn size") is None

    def test_payload_round_trips_into_the_response_model(self):
        """Pins the tool's JSON keys against the Pydantic contract the frontend
        consumes, so a rename on either side fails here first."""
        payload = json.loads(
            json.dumps(
                {
                    "product_id": "12",
                    "product_name": "Yonex Astrox 99 Game",
                    "currency": "VND",
                    "options": [
                        {
                            "variant_id": "45",
                            "size": "4U",
                            "color": "Đỏ",
                            "unit_price": 2689000.0,
                            "orderable": 7,
                            "branches": [
                                {
                                    "store_id": 2,
                                    "store_name": "Q7",
                                    "quantity": 3,
                                }
                            ],
                        }
                    ],
                }
            )
        )

        resp = ChatResponse(answer="x", sources=[], order_selection=payload)

        assert isinstance(resp.order_selection, OrderSelection)
        option = resp.order_selection.options[0]
        assert option.variant_id == "45"
        assert option.orderable == 7
        assert option.branches[0].store_name == "Q7"
