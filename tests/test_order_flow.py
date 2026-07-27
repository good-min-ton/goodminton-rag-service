from app.services.order_flow import (
    BROWSING,
    ORDER_CONFIRMED,
    WAITING_CONFIRMATION,
    next_order_status,
    order_directive,
    suppresses_recommendations,
)
from app.services.query_understanding import QueryUnderstanding


def _qu(categories=None):
    return QueryUnderstanding(categories=categories or [], retrieval_query="x")


def test_browsing_plus_draft_enters_waiting():
    assert (
        next_order_status(
            BROWSING, _qu(), order_draft_emitted=True, order_placed_id=None
        )
        == WAITING_CONFIRMATION
    )


def test_waiting_plus_placed_id_confirms():
    assert (
        next_order_status(
            WAITING_CONFIRMATION, _qu(), order_draft_emitted=False, order_placed_id=55
        )
        == ORDER_CONFIRMED
    )


def test_waiting_plus_new_category_resets_to_browsing():
    assert (
        next_order_status(
            WAITING_CONFIRMATION,
            _qu(["Áo cầu lông"]),
            order_draft_emitted=False,
            order_placed_id=None,
        )
        == BROWSING
    )


def test_confirmed_plus_new_category_resets_to_browsing():
    assert (
        next_order_status(
            ORDER_CONFIRMED,
            _qu(["Giày cầu lông"]),
            order_draft_emitted=False,
            order_placed_id=None,
        )
        == BROWSING
    )


def test_new_category_while_already_browsing_stays_browsing():
    # rule 1 only fires when current != BROWSING; a category in BROWSING is normal browsing
    assert (
        next_order_status(
            BROWSING,
            _qu(["Quần cầu lông"]),
            order_draft_emitted=False,
            order_placed_id=None,
        )
        == BROWSING
    )


def test_draft_outranks_stale_placed_id():
    # new order started after a prior placement (stale id resent) -> WAITING, not CONFIRMED
    assert (
        next_order_status(
            ORDER_CONFIRMED, _qu(), order_draft_emitted=True, order_placed_id=99
        )
        == WAITING_CONFIRMATION
    )


def test_waiting_no_signal_stays_waiting():
    assert (
        next_order_status(
            WAITING_CONFIRMATION, _qu(), order_draft_emitted=False, order_placed_id=None
        )
        == WAITING_CONFIRMATION
    )


def test_none_current_defaults_browsing():
    assert (
        next_order_status(None, _qu(), order_draft_emitted=False, order_placed_id=None)
        == BROWSING
    )


def test_suppression_flags():
    assert suppresses_recommendations(WAITING_CONFIRMATION) is True
    assert suppresses_recommendations(ORDER_CONFIRMED) is True
    assert suppresses_recommendations(BROWSING) is False
    assert suppresses_recommendations(None) is False


def test_directive_text():
    assert "xác nhận" in order_directive(WAITING_CONFIRMATION).lower()
    assert order_directive(BROWSING) == ""
    assert "đặt" in order_directive(ORDER_CONFIRMED).lower()


def test_draft_outranks_new_category_same_turn():
    # From a non-BROWSING state, a turn that BOTH names a category AND emits a
    # draft is an active order -> WAITING (suppress), not BROWSING (leak).
    assert (
        next_order_status(
            WAITING_CONFIRMATION,
            _qu(["Áo cầu lông"]),
            order_draft_emitted=True,
            order_placed_id=None,
        )
        == WAITING_CONFIRMATION
    )
