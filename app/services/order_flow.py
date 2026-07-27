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
    if order_draft_emitted:  # a fresh draft = an active order -> always WAITING
        return WAITING_CONFIRMATION
    if qu.categories and cur != BROWSING:  # new browse resets order context
        return BROWSING
    if order_placed_id is not None:
        return ORDER_CONFIRMED
    return cur


def suppresses_recommendations(status: str | None) -> bool:
    return status in _SUPPRESS


def order_directive(status: str | None) -> str:
    return _DIRECTIVES.get(status or BROWSING, "")
