import pytest
from eval.rerank_retriever import RerankRetriever
from eval.rerank_client import RerankUnavailable


class _Base:
    def __init__(self, ids):
        self._ids = ids
        self.calls = []

    async def retrieve(self, query, k):
        self.calls.append(k)
        return self._ids[:k]


class _Conn:
    def __init__(self, texts):
        self._texts = texts

    async def fetch(self, sql, ids):
        return [
            {"source_id": i, "text": self._texts[i]} for i in ids if i in self._texts
        ]


class _Pool:
    def __init__(self, texts):
        self._c = _Conn(texts)

    def acquire(self):
        c = self._c

        class _Ctx:
            async def __aenter__(self):
                return c

            async def __aexit__(self, *a):
                return False

        return _Ctx()


class _RC:
    def __init__(self, scores, fail=False):
        self._scores = scores
        self.fail = fail
        self.seen = None

    async def rerank(self, query, documents):
        if self.fail:
            raise RerankUnavailable("boom")
        self.seen = documents
        return self._scores


@pytest.mark.asyncio
async def test_rerank_reorders_by_score():
    base = _Base(["p1", "p2", "p3"])
    pool = _Pool({"p1": "t1", "p2": "t2", "p3": "t3"})
    rc = _RC([0.1, 0.9, 0.5])  # p2 best, then p3, then p1
    out = await RerankRetriever(base, rc, pool, candidate_n=3).retrieve("q", 2)
    assert out == ["p2", "p3"]
    assert base.calls == [3]  # fetched candidate_n
    assert rc.seen == ["t1", "t2", "t3"]  # texts in base order


@pytest.mark.asyncio
async def test_rerank_degrades_to_base_order_on_failure():
    base = _Base(["p1", "p2", "p3"])
    pool = _Pool({"p1": "t1", "p2": "t2", "p3": "t3"})
    out = await RerankRetriever(base, _RC([], fail=True), pool, candidate_n=3).retrieve(
        "q", 2
    )
    assert out == ["p1", "p2"]  # base order preserved
