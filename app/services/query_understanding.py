"""Deterministic query understanding: extract intent/category/price from the
message (rule-based first, LLM fallback), and contextualize elliptical follow-ups
using prior conversation state. Category selection is NEVER left to the tool loop.
"""

import logging
import re

from pydantic import BaseModel, Field

from app.services.conversation_state import ConversationState

log = logging.getLogger(__name__)

# KEY = the exact `product['category']` string stored in kb_chunks.metadata.
# VALUE = trigger keywords, matched on WORD BOUNDARIES (see _CATEGORY_PATTERNS).
#
# Substring matching used to be the rule here and it mis-tagged the shop's most
# common question: "bao nhiêu" contains "ao", so every "vợt ... giá bao nhiêu?"
# was classified as a shirt question as well as a racket one. Multi-category
# retrieval then reserved a fixed quota of shirt chunks, which is how a racket
# question ended up showing shirts among its product cards.
#
# Bare unaccented "ao" and "quan" stay out: they are ordinary Vietnamese words
# ("ao", "quan tâm", "liên quan") that a word boundary alone cannot disambiguate,
# so they only count inside the full phrase. The accented forms are unambiguous.
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Vợt cầu lông": ["vợt", "vot", "racket", "racquet"],
    "Giày cầu lông": ["giày", "giay", "shoe", "sneaker"],
    "Quần cầu lông": ["quần", "quan cầu lông", "quan cau long", "short", "shorts"],
    "Áo cầu lông": ["áo", "ao cầu lông", "ao cau long", "shirt", "jersey"],
    "Dây cước cầu lông": [
        "dây cước",
        "day cuoc",
        "cước",
        "cuoc",
        "string",
        "căng vợt",
        "cang vot",
    ],
}

# `\b` is Unicode-aware for str patterns, so it treats Vietnamese diacritics as
# word characters: "ao" does not match inside "bao", and "áo" still matches on
# its own. Compiled once — analyze() runs on every chat turn.
_CATEGORY_PATTERNS: dict[str, re.Pattern[str]] = {
    category: re.compile("|".join(rf"\b{re.escape(kw)}\b" for kw in keywords))
    for category, keywords in CATEGORY_KEYWORDS.items()
}

_PRICE_CHEAPEST = [
    "rẻ nhất",
    "re nhat",
    "rẻ",
    "re ",
    "giá thấp",
    "gia thap",
    "rẻ hơn",
    "re hon",
]
_INTENT_PRICE = ["giá", "gia ", "bao nhiêu", "bao nhieu", "giảm giá", "sale"]
_INTENT_STOCK = [
    "còn hàng",
    "con hang",
    "hết hàng",
    "het hang",
    "size",
    "tồn kho",
    "ton kho",
]
_INTENT_BUY = ["mua", "đặt", "dat ", "order", "chốt", "lấy"]

# Shopping-refinement signals not already covered by category/price/intent keywords.
_REFINE_KEYWORDS = [
    "cái nào",
    "loại nào",
    "mẫu nào",
    "mẫu",
    "màu",
    "cỡ",
    "nhẹ",
    "nặng",
    "bền",
    "cứng",
    "dẻo",
    "êm",
    "đắt",
    "tốt",
    "xịn",
    "cao cấp",
    "phù hợp",
    "gợi ý",
    "tư vấn",
    "sản phẩm",
]


class QueryUnderstanding(BaseModel):
    intent: str | None = None
    categories: list[str] = Field(default_factory=list)
    price_preference: str | None = None
    retrieval_query: str = ""
    product_query: bool = True


class QueryUnderstandingService:
    def __init__(self, llm) -> None:
        self._llm = llm  # LLMService

    def _rule_categories(self, low: str) -> list[str]:
        return [c for c, pat in _CATEGORY_PATTERNS.items() if pat.search(low)]

    def _rule_intent(self, low: str) -> str | None:
        if any(k in low for k in _INTENT_STOCK):
            return "stock"
        if any(k in low for k in _INTENT_PRICE):
            return "price"
        if any(k in low for k in _INTENT_BUY):
            return "buy"
        return "browse"

    async def _llm_category(self, message: str) -> list[str]:
        prompt = (
            "Phân loại câu sau vào MỘT danh mục sản phẩm cầu lông, trả lời DUY NHẤT "
            "một trong: Vợt cầu lông | Giày cầu lông | Quần cầu lông | Áo cầu lông | "
            "Dây cước cầu lông | KHÔNG RÕ. Câu: " + message
        )
        try:
            raw = await self._llm.chat([{"role": "user", "content": prompt}])
        except Exception as exc:
            log.warning("LLM category fallback failed (%s)", exc)
            return []
        if not isinstance(raw, str):
            return []
        raw = raw.strip()
        for category in CATEGORY_KEYWORDS:
            if category.lower() in raw.lower():
                return [category]
        return []

    async def analyze(
        self, message: str, state: ConversationState
    ) -> QueryUnderstanding:
        low = message.lower()
        categories = self._rule_categories(low)
        intent = self._rule_intent(low)
        price_pref = "cheapest" if any(k in low for k in _PRICE_CHEAPEST) else None
        product_query = (
            bool(self._rule_categories(low))
            or price_pref is not None
            or intent in {"price", "stock", "buy"}
            or any(k in low for k in _REFINE_KEYWORDS)
        )

        if not categories:
            # LLM fallback for category only; if it also fails, inherit from state.
            categories = await self._llm_category(message)
        if not categories and state.categories:
            categories = list(state.categories)  # elliptical follow-up inherits scope

        # Contextualize the retrieval query: append inherited scope so the vector
        # search is anchored even when the message itself is elliptical ("rẻ nhất").
        retrieval_query = message.strip()
        if categories and not self._rule_categories(low):
            retrieval_query = f"{message.strip()} {' '.join(categories)}"

        return QueryUnderstanding(
            intent=intent,
            categories=categories,
            price_preference=price_pref,
            retrieval_query=retrieval_query,
            product_query=product_query,
        )
