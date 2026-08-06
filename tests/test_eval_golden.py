import json
import pytest
from eval.golden import load_golden, GoldenRecord


def _write(tmp_path, rows):
    p = tmp_path / "g.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    return str(p)


def _row(**kw):
    base = {
        "id": "q1",
        "query": "vợt nhẹ",
        "query_type": "attribute",
        "relevant_source_ids": ["1"],
        "expected_categories": ["Vợt cầu lông"],
        "price_constrained": False,
        "source": "hand",
        "notes": "",
    }
    base.update(kw)
    return base


def test_loads_valid_record(tmp_path):
    kept, excluded = load_golden(_write(tmp_path, [_row()]))
    assert excluded == []
    assert kept[0] == GoldenRecord(
        id="q1",
        query="vợt nhẹ",
        query_type="attribute",
        relevant_source_ids=["1"],
        expected_categories=["Vợt cầu lông"],
        price_constrained=False,
        source="hand",
        notes="",
    )


def test_rejects_out_of_vocab_category(tmp_path):
    path = _write(tmp_path, [_row(expected_categories=["vot-cau-long"])])
    with pytest.raises(ValueError, match="category"):
        load_golden(path)


def test_rejects_bad_query_type(tmp_path):
    path = _write(tmp_path, [_row(query_type="price")])
    with pytest.raises(ValueError, match="query_type"):
        load_golden(path)


def test_rejects_bad_source(tmp_path):
    path = _write(tmp_path, [_row(source="bogus")])
    with pytest.raises(ValueError, match="source"):
        load_golden(path)


def test_rejects_empty_relevant_ids(tmp_path):
    path = _write(tmp_path, [_row(relevant_source_ids=[])])
    with pytest.raises(ValueError, match="relevant_source_ids"):
        load_golden(path)


def test_drops_record_with_no_relevant_in_corpus(tmp_path):
    path = _write(tmp_path, [_row(id="q9", relevant_source_ids=["999"])])
    kept, excluded = load_golden(path, valid_source_ids={"1", "2"})
    assert kept == []
    assert excluded == [{"id": "q9", "reason": "no relevant in corpus"}]


def test_filters_missing_ids_but_keeps_present(tmp_path):
    path = _write(tmp_path, [_row(relevant_source_ids=["1", "999"])])
    kept, _ = load_golden(path, valid_source_ids={"1"})
    assert kept[0].relevant_source_ids == ["1"]


def test_accepts_known_item_query_type(tmp_path):
    row = {
        "id": "k1",
        "query": "vợt nhẹ khung carbon",
        "query_type": "known-item",
        "relevant_source_ids": ["1"],
        "expected_categories": ["Vợt cầu lông"],
        "price_constrained": False,
        "source": "hand",
        "notes": "",
    }
    p = tmp_path / "g.jsonl"
    p.write_text(json.dumps(row, ensure_ascii=False))
    kept, _ = load_golden(str(p))
    assert kept[0].query_type == "known-item"
