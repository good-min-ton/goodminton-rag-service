"""Deterministic order-flow state machine for the chat endpoint. No I/O, no LLM."""

from app.services.query_understanding import QueryUnderstanding

BROWSING = "BROWSING"
WAITING_CONFIRMATION = "WAITING_CONFIRMATION"
ORDER_CONFIRMED = "ORDER_CONFIRMED"

_SUPPRESS = {WAITING_CONFIRMATION, ORDER_CONFIRMED}

_DIRECTIVES = {
    WAITING_CONFIRMATION: (
        "\n\nKhách đang có bảng chọn / thẻ đơn hàng mở trên màn hình và đang thao "
        "tác trên đó. CHỈ hỗ trợ: giải đáp về sản phẩm đang chọn, hủy, hoặc mở lại "
        "bảng chọn. TUYỆT ĐỐI KHÔNG gợi ý sản phẩm mới, không liệt kê sản phẩm khác, "
        "và KHÔNG hỏi size/màu/số lượng bằng chữ — khách bấm chọn trên bảng."
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
    order_card_emitted: bool,
    order_just_placed: bool,
) -> str:
    """`order_card_emitted` is true when this turn put an order card on screen.

    That used to mean a priced draft from prepare_order. It now means the variant
    picker: the customer is mid-order from the moment they can choose, not from
    the moment a total exists, so the "stop recommending other products" rule has
    to start there too.
    """
    cur = current or BROWSING
    if order_card_emitted:  # an order is in progress -> always WAITING
        return WAITING_CONFIRMATION
    if qu.categories and cur != BROWSING:  # new browse resets order context
        return BROWSING
    if order_just_placed:
        return ORDER_CONFIRMED
    return cur


def suppresses_recommendations(status: str | None) -> bool:
    return status in _SUPPRESS


def order_directive(status: str | None) -> str:
    return _DIRECTIVES.get(status or BROWSING, "")
