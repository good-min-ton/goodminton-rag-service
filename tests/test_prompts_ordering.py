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


def test_prompt_offers_no_way_to_order_without_the_picker():
    """The model has no path to a variant_id any more: prepare_order is gone and
    the prompt must not describe one, or a stuck model will invent it."""
    assert "prepare_order" not in SYSTEM_PROMPT
    assert "variant_id" not in SYSTEM_PROMPT


def test_prompt_forbids_markup_the_bubble_cannot_render():
    """The bubble renders a deliberately small Markdown subset. Anything outside
    it degrades to stripped text, which is how a customer asking to switch racket
    versions got a bare "!Bảng chọn sản phẩm" under the answer: the model emitted
    an image placeholder for the picker the app was already rendering."""
    for keyword in (
        "![mô tả](url)",
        "KHÔNG tự vẽ hay mô tả giao diện",
        "bảng chọn sản phẩm và thẻ sản phẩm do ứng dụng tự hiển thị",
    ):
        assert keyword in SYSTEM_PROMPT, keyword


def test_system_prompt_still_formats_with_context():
    # No stray braces were introduced; .format still works.
    assert "seeded-context-marker" in SYSTEM_PROMPT.format(
        context="seeded-context-marker"
    )
