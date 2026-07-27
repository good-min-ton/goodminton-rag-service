from unittest.mock import AsyncMock

from app.services.conversation_state import ConversationState
from app.services.query_understanding import QueryUnderstandingService


def _svc(llm=None):
    return QueryUnderstandingService(llm or AsyncMock())


async def test_single_category_pants():
    qu = await _svc().analyze("mua quần cầu lông", ConversationState())
    assert qu.categories == ["Quần cầu lông"]
    assert qu.retrieval_query == "mua quần cầu lông"


async def test_multi_category_pants_and_shoes():
    qu = await _svc().analyze("cho tôi xem quần và giày", ConversationState())
    assert set(qu.categories) == {"Quần cầu lông", "Giày cầu lông"}


async def test_elliptical_inherits_categories_from_state():
    state = ConversationState(categories=["Quần cầu lông"])
    qu = await _svc().analyze("rẻ nhất", state)
    assert qu.categories == ["Quần cầu lông"]
    assert qu.price_preference == "cheapest"
    # contextualized query carries the inherited scope so the vector search is scoped
    assert "Quần cầu lông" in qu.retrieval_query


async def test_price_preference_cheapest_keywords():
    qu = await _svc().analyze("quần nào rẻ nhất", ConversationState())
    assert qu.price_preference == "cheapest"


async def test_no_rule_match_falls_back_to_llm():
    llm = AsyncMock()
    llm.chat.return_value = "Giày cầu lông"  # LLM returns a category label
    qu = await _svc(llm).analyze("đôi nào bền cho người mới", ConversationState())
    assert qu.categories == ["Giày cầu lông"]
    llm.chat.assert_awaited_once()


async def test_llm_fallback_unusable_output_yields_no_category():
    llm = AsyncMock()
    llm.chat.return_value = "tôi không rõ"
    qu = await _svc(llm).analyze("asdfqwer", ConversationState())
    assert qu.categories == []  # unfiltered retrieval downstream (current behavior)


async def test_product_query_true_for_buy_intent():
    qu = await _svc().analyze("tôi muốn mua quần cầu lông", ConversationState())
    assert qu.product_query is True


async def test_product_query_true_for_price_preference():
    qu = await _svc().analyze("loại nào rẻ nhất?", ConversationState())
    assert qu.product_query is True


async def test_product_query_true_for_refine_keyword():
    qu = await _svc().analyze("cái nào nhẹ hơn?", ConversationState())
    assert qu.product_query is True


async def test_product_query_true_for_category_and_price_intent():
    qu = await _svc().analyze("vợt astrox giá bao nhiêu", ConversationState())
    assert qu.product_query is True


async def test_product_query_false_for_weather_chitchat():
    qu = await _svc().analyze("hôm nay trời đẹp nhỉ", ConversationState())
    assert qu.product_query is False


async def test_product_query_false_for_policy_question():
    qu = await _svc().analyze("chính sách bảo hành thế nào?", ConversationState())
    assert qu.product_query is False


async def test_product_query_false_for_thanks():
    qu = await _svc().analyze("cảm ơn shop nhé", ConversationState())
    assert qu.product_query is False


async def test_product_query_false_for_offtopic_followup_with_inherited_category():
    state = ConversationState(categories=["Quần cầu lông"])
    qu = await _svc().analyze("chính sách bảo hành thế nào?", state)
    # Categories are still inherited for retrieval scoping...
    assert qu.categories == ["Quần cầu lông"]
    # ...but the turn itself is not a product query, so cards must be gated off.
    assert qu.product_query is False
