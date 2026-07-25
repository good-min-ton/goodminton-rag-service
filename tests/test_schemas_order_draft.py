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
