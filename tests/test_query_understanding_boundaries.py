"""Category keywords must match whole words, not substrings.

Substring matching tagged the shop's most common question wrong: "bao nhiêu"
contains "ao", so "vợt ... giá bao nhiêu?" came back as both a racket and a
shirt question. Multi-category retrieval then reserved a quota of shirt chunks,
which is how shirts reached the product cards of a racket question.
"""

import pytest

from app.services.query_understanding import QueryUnderstandingService


@pytest.fixture
def svc():
    return QueryUnderstandingService(llm=None)


@pytest.mark.parametrize(
    "message",
    [
        "Vợt Astrox 99 giá bao nhiêu?",
        "Vợt này bao nhiêu vậy shop?",
        "Cho mình xem vợt Yonex giá bao nhiêu tiền",
        "Mua vợt bao nhiêu tiền thì có bảo hành",
    ],
    ids=["price", "short-price", "long-price", "warranty"],
)
def test_bao_nhieu_does_not_add_shirts(svc, message):
    assert svc._rule_categories(message.lower()) == ["Vợt cầu lông"]


@pytest.mark.parametrize(
    "message",
    ["Mình quan tâm tới vợt công thủ", "Vấn đề liên quan tới vợt của mình"],
    ids=["quan-tam", "lien-quan"],
)
def test_quan_inside_a_word_does_not_add_pants(svc, message):
    assert svc._rule_categories(message.lower()) == ["Vợt cầu lông"]


def test_bao_nhieu_does_not_add_shirts_to_a_shoe_question(svc):
    assert svc._rule_categories("giày cầu lông size 42 bao nhiêu?") == ["Giày cầu lông"]


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Áo cầu lông nam size L", "Áo cầu lông"),
        ("Quần cầu lông có size XL không", "Quần cầu lông"),
        ("Cho mình xem giày cầu lông", "Giày cầu lông"),
        ("Dây cước nào bền nhất", "Dây cước cầu lông"),
        ("Cây vợt này nặng bao nhiêu gram", "Vợt cầu lông"),
    ],
)
def test_real_category_questions_still_match(svc, message, expected):
    assert svc._rule_categories(message.lower()) == [expected]


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("ao cau long nam gia bao nhieu", "Áo cầu lông"),
        ("quan cau long gia bao nhieu", "Quần cầu lông"),
        ("vot cau long cho nguoi moi", "Vợt cầu lông"),
        ("giay cau long size 42", "Giày cầu lông"),
    ],
)
def test_unaccented_input_still_matches(svc, message, expected):
    """Bare "ao"/"quan" are ordinary words, so they only count inside the full
    phrase — someone typing without diacritics still gets the right category."""
    assert svc._rule_categories(message) == [expected]


def test_genuinely_multi_category_question_keeps_both(svc):
    """The fix must not over-correct: a question that really is about two
    categories still returns both."""
    assert set(svc._rule_categories("mua vợt tặng áo không shop?")) == {
        "Vợt cầu lông",
        "Áo cầu lông",
    }
