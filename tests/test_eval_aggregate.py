import pytest
from eval.golden import GoldenRecord
from eval.aggregate import evaluate_retriever, category_accuracy


def _rec(
    id,
    query_type="browse",
    relevant=("p1",),
    cats=("Vợt cầu lông",),
    price=False,
    source="hand",
):
    return GoldenRecord(
        id, f"q{id}", query_type, list(relevant), list(cats), price, source, ""
    )


class _FakeRetriever:
    def __init__(self, mapping):
        self._mapping = mapping  # query -> ranked list

    async def retrieve(self, query, k):
        return self._mapping[query][:k]


@pytest.mark.asyncio
async def test_evaluate_excludes_price_and_slices():
    records = [
        _rec("1", relevant=("p1",)),
        _rec("2", query_type="spec", relevant=("p2",)),
        _rec("3", price=True, relevant=("p9",)),  # excluded from pooled
    ]
    mapping = {"q1": ["p1"], "q2": ["x", "p2"], "q3": ["nope"]}
    block = await evaluate_retriever(_FakeRetriever(mapping), records, [5, 10])
    assert block["n_queries"] == 2  # price query excluded
    assert block["price_bucket"]["n"] == 1
    assert block["price_bucket"]["recall@10"] == 0.0
    assert block["pooled"]["recall@10"][0] == pytest.approx(1.0)  # both found
    assert "browse" in block["per_query_type"] and "spec" in block["per_query_type"]


@pytest.mark.asyncio
async def test_category_accuracy_uses_sets():
    records = [_rec("1", cats=("Vợt cầu lông",))]

    class _QU:
        categories = ["Vợt cầu lông"]

    async def analyze(query):
        return _QU()

    cat = await category_accuracy(analyze, records)
    assert cat["exact_match"][0] == 1.0
    assert cat["precision"][0] == 1.0
