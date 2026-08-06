"""Cross-encoder reranker over a base retriever's top-N candidates.
Experiment lives in eval/ (app/ retrieval unchanged)."""

import logging

from eval.rerank_client import RerankUnavailable

log = logging.getLogger(__name__)

_TEXT_SQL = (
    "SELECT source_id, string_agg(content, ' ' ORDER BY chunk_index) AS text "
    "FROM kb_chunks WHERE doc_type='product' AND source_id = ANY($1::text[]) "
    "GROUP BY source_id"
)


class RerankRetriever:
    def __init__(self, base_retriever, rerank_client, pool, candidate_n: int = 30):
        self._base = base_retriever
        self._rerank = rerank_client
        self._pool = pool
        self._candidate_n = candidate_n

    async def _texts(self, ids: list[str]) -> dict[str, str]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(_TEXT_SQL, ids)
        return {str(r["source_id"]): r["text"] for r in rows}

    async def retrieve(self, query: str, k: int) -> list[str]:
        cands = await self._base.retrieve(query, self._candidate_n)
        if not cands:
            return []
        text_map = await self._texts(cands)
        docs = [text_map.get(c, "") for c in cands]
        try:
            scores = await self._rerank.rerank(query, docs)
        except RerankUnavailable as exc:
            log.warning("rerank failed (%s); falling back to base order", exc)
            return cands[:k]
        order = sorted(range(len(cands)), key=lambda i: (-scores[i], i))
        return [cands[i] for i in order][:k]
