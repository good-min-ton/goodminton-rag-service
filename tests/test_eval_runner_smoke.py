import pytest
from eval.golden import GoldenRecord
from eval.runner import build_agg


class _FakeRetriever:
    async def retrieve(self, query, k):
        return ["p1"]


@pytest.mark.asyncio
async def test_build_agg_assembles_report_shape(tmp_path):
    records = [
        GoldenRecord("1", "vợt", "browse", ["p1"], ["Vợt cầu lông"], False, "hand", "")
    ]

    class _QU:
        categories = ["Vợt cầu lông"]

    async def analyze(query):
        return _QU()

    agg = await build_agg(
        records=records,
        excluded=[{"id": "9", "reason": "no relevant in corpus"}],
        valid_ids={"p1"},
        retrievers={"production": _FakeRetriever()},
        qu_analyze=analyze,
        label="t",
        k_values=[5, 10],
    )
    assert agg["label"] == "t"
    assert agg["retrievers"]["production"]["n_excluded"] == 1
    assert agg["corpus"]["count"] == 1
    assert agg["category"]["exact_match"][0] == 1.0
