# tests/test_eval_retriever.py
from dataclasses import dataclass
import pytest
from eval.retriever import NaiveVectorRetriever, ProductionRetriever, OVER_FETCH_FACTOR


@dataclass
class _C:
    source_id: str
    chunk_index: int
    distance: float


class _FakeEmbedder:
    def __init__(self):
        self.embedded: list[str] = []

    async def embed(self, text):
        self.embedded.append(text)
        return [0.0, 1.0]


class _FakeRetrieval:
    def __init__(self, chunks):
        self._chunks = chunks
        self.calls: list[dict] = []

    async def search(self, query_embedding, k=None, categories=None, doc_type=None):
        self.calls.append({"k": k, "categories": categories, "doc_type": doc_type})
        return self._chunks


class _FakeQU:
    def __init__(self, retrieval_query, categories):
        self._rq = retrieval_query
        self._cats = categories

    async def analyze(self, message, state):
        from app.services.query_understanding import QueryUnderstanding

        return QueryUnderstanding(retrieval_query=self._rq, categories=self._cats)


@pytest.mark.asyncio
async def test_naive_embeds_raw_query_and_filters_products():
    emb = _FakeEmbedder()
    ret = _FakeRetrieval([_C("p1", 0, 0.1), _C("p1", 1, 0.2), _C("p2", 0, 0.15)])
    out = await NaiveVectorRetriever(emb, ret).retrieve("vợt rẻ", 2)
    assert out == ["p1", "p2"]
    assert emb.embedded == ["vợt rẻ"]  # RAW query
    assert ret.calls[0] == {
        "k": 2 * OVER_FETCH_FACTOR,
        "categories": None,
        "doc_type": "product",
    }


@pytest.mark.asyncio
async def test_production_uses_retrieval_query_and_categories():
    emb = _FakeEmbedder()
    ret = _FakeRetrieval([_C("p2", 0, 0.15), _C("p1", 0, 0.1)])
    qu = _FakeQU(retrieval_query="rẻ nhất Vợt cầu lông", categories=["Vợt cầu lông"])
    out = await ProductionRetriever(emb, ret, qu).retrieve("rẻ nhất", 2)
    assert out == ["p1", "p2"]  # globally re-ranked by distance
    assert emb.embedded == ["rẻ nhất Vợt cầu lông"]  # CONTEXTUALIZED query
    assert ret.calls[0]["categories"] == ["Vợt cầu lông"]
    assert ret.calls[0]["doc_type"] is None  # mirrors chat.py (no doc_type filter)


@pytest.mark.asyncio
async def test_production_no_categories_passes_none():
    emb = _FakeEmbedder()
    ret = _FakeRetrieval([_C("p1", 0, 0.1)])
    qu = _FakeQU(retrieval_query="chào", categories=[])
    await ProductionRetriever(emb, ret, qu).retrieve("chào", 1)
    assert ret.calls[0]["categories"] is None
