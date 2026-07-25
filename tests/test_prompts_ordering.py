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
