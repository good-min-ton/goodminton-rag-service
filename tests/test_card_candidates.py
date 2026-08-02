from app.routers.chat import _card_candidates
from app.services.retrieval import Chunk


def _p(sid, dist, name="Vợt X"):
    return Chunk("product", sid, 0, f"Sản phẩm: {name}", dist)


def test_distance_threshold_drops_weak_chunks():
    chunks = [_p("1", 0.40), _p("2", 0.90)]
    out = _card_candidates(chunks, [], 0.62)
    assert [c["id"] for c in out] == ["1"]


def test_threshold_disabled_when_non_positive():
    chunks = [_p("1", 0.40), _p("2", 0.90)]
    out = _card_candidates(chunks, [], 0.0)
    assert [c["id"] for c in out] == ["1", "2"]


def test_tool_products_first_and_deduped():
    chunks = [_p("5", 0.30)]
    tool = [{"id": "9", "name": "Nine"}, {"id": "5", "name": "Five"}]
    out = _card_candidates(chunks, tool, 0.62)
    # tool products lead; chunk 5 is deduped against tool product 5
    assert [c["id"] for c in out] == ["9", "5"]


def test_non_digit_and_static_skipped():
    chunks = [_p("abc", 0.30), Chunk("static", "policy.md", 0, "x", 0.1)]
    out = _card_candidates(chunks, [], 0.62)
    assert out == []


def test_name_extracted_from_chunk():
    out = _card_candidates([_p("1", 0.3, name="Yonex Astrox 99 Pro")], [], 0.62)
    assert out[0]["name"] == "Yonex Astrox 99 Pro"
