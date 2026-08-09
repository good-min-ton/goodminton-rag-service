from app.core.prompts import SYSTEM_PROMPT


def test_system_prompt_has_ordering_rules():
    """Rules the ordering flow depends on. Each one exists because dropping it
    produced a real failure: a wrong variant, a promised stock the checkout could
    not draw on, or a customer told their order was placed when it was not."""
    for keyword in (
        # Buying opens the picker; the model no longer picks the variant.
        "start_order(product_id)",
        "TUYỆT ĐỐI KHÔNG hỏi size, màu hay số lượng trước",
        "TUYỆT ĐỐI không bịa product_id",
        # Orderable stock is the central store's, never a cross-store total.
        "orderable",
        "branches",
        "KHÔNG cộng vào `orderable`",
        "KHÔNG hứa giữ hàng",
        # The card owns money and address; the bot must not pre-empt either.
        "KHÔNG nêu tổng tiền",
        "KHÔNG nói đơn đã được đặt",
        "KHÔNG hỏi địa chỉ giao hàng trong chat",
        # Price and stock still come from a tool, never from retrieved context.
        "get_product_availability(product_id)",
        "KHÔNG ĐƯỢC TRẢ LỜI GIÁ TỪ CONTEXT",
    ):
        assert keyword in SYSTEM_PROMPT, keyword


def test_prepare_order_is_described_as_the_fallback():
    """It stays available for a customer who states size, colour and quantity
    outright, but the picker is the default: a chosen variant cannot be the
    wrong one."""
    assert "prepare_order(items)" in SYSTEM_PROMPT
    assert "ĐƯỜNG LUI" in SYSTEM_PROMPT


def test_system_prompt_still_formats_with_context():
    # No stray braces were introduced; .format still works.
    assert "seeded-context-marker" in SYSTEM_PROMPT.format(
        context="seeded-context-marker"
    )
