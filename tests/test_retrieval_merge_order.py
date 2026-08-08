"""The multi-category quota merge must return globally distance-sorted chunks.

Callers slice this list: _prepare_chat_pipeline feeds `chunks[:retrieval_top_k]`
to the LLM and the whole list to card selection. In per-category order that slice
is simply the first category's chunks, so the second category could reach the
product cards without ever reaching the answer those cards sit under.

_search_one is stubbed, so this needs no database.
"""

import pytest

from app.services.retrieval import Chunk, RetrievalService


def _chunk(source_id: str, distance: float) -> Chunk:
    return Chunk(
        doc_type="product",
        source_id=source_id,
        chunk_index=0,
        content=f"P{source_id}",
        distance=distance,
    )


class _StubbedRetrieval(RetrievalService):
    """Returns a canned per-category result, recording the requested limits."""

    def __init__(self, by_category: dict[str, list[Chunk]]):
        super().__init__(pool=None)  # type: ignore[arg-type]
        self._by_category = by_category
        self.requested_limits: list[int] = []

    async def _search_one(self, query_embedding, top_k, categories, doc_type):
        self.requested_limits.append(top_k)
        cat = categories[0] if categories else None
        return self._by_category.get(cat, [])[:top_k]


@pytest.mark.asyncio
async def test_merged_categories_come_back_sorted_by_distance():
    svc = _StubbedRetrieval(
        {
            # The nearer chunk overall sits in the SECOND category.
            "Vợt cầu lông": [_chunk("10", 0.30), _chunk("11", 0.40)],
            "Áo cầu lông": [_chunk("20", 0.10), _chunk("21", 0.35)],
        }
    )

    chunks = await svc.search([0.0], k=4, categories=["Vợt cầu lông", "Áo cầu lông"])

    assert [c.source_id for c in chunks] == ["20", "10", "21", "11"]
    assert [c.distance for c in chunks] == sorted(c.distance for c in chunks)


@pytest.mark.asyncio
async def test_each_category_still_gets_its_quota():
    """Sorting must not undo the quota: the point of the merge is that a category
    with uniformly worse distances is still represented."""
    svc = _StubbedRetrieval(
        {
            "Vợt cầu lông": [_chunk(str(i), 0.10 + i / 100) for i in range(10)],
            "Giày cầu lông": [_chunk("90", 0.80), _chunk("91", 0.85)],
        }
    )

    chunks = await svc.search([0.0], k=4, categories=["Vợt cầu lông", "Giày cầu lông"])

    assert svc.requested_limits == [2, 2]  # k // number of categories
    assert {c.source_id for c in chunks} >= {"90"}
    assert len(chunks) == 4


@pytest.mark.asyncio
async def test_single_category_is_left_to_the_database_ordering():
    """One category means one query, already ORDER BY distance in SQL — no merge,
    no client-side re-sort."""
    svc = _StubbedRetrieval({"Vợt cầu lông": [_chunk("10", 0.30), _chunk("11", 0.40)]})

    chunks = await svc.search([0.0], k=5, categories=["Vợt cầu lông"])

    assert [c.source_id for c in chunks] == ["10", "11"]
    assert svc.requested_limits == [5]
