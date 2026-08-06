import pytest
from eval.golden import GoldenRecord
from eval.compare import compare, render_compare_markdown


def _rec(id, qtype="known-item", relevant=("p1",), price=False):
    return GoldenRecord(
        id, f"q{id}", qtype, list(relevant), ["Vợt cầu lông"], price, "hand", ""
    )


class _R:
    def __init__(self, mapping):
        self._m = mapping

    async def retrieve(self, query, k):
        return self._m[query][:k]


@pytest.mark.asyncio
async def test_compare_reports_positive_delta_per_slice():
    records = [_rec("1", relevant=("p1",)), _rec("2", relevant=("p2",))]
    baseline = _R({"q1": ["x", "p1"], "q2": ["x", "p2"]})
    cand = _R({"q1": ["p1"], "q2": ["p2"]})
    res = await compare(records, baseline, cand, k=10)
    pooled = res["groups"]["pooled"]
    assert pooled["mrr"]["baseline"] == pytest.approx(0.5)
    assert pooled["mrr"]["cand"] == pytest.approx(1.0)
    assert pooled["mrr"]["delta"][0] == pytest.approx(0.5)  # cand - baseline
    assert res["groups"]["known-item"]["n"] == 2
    md = render_compare_markdown(res, "t")
    assert "known-item" in md and "mrr" in md


@pytest.mark.asyncio
async def test_compare_excludes_price_and_slices_by_type():
    records = [
        _rec("1", qtype="known-item"),
        _rec("2", qtype="browse"),
        _rec("3", price=True),
    ]
    r = _R({"q1": ["p1"], "q2": ["p1"], "q3": ["p1"]})
    res = await compare(records, r, r, k=10)
    assert res["groups"]["pooled"]["n"] == 2
    assert set(res["groups"]) == {"pooled", "known-item", "browse"}
