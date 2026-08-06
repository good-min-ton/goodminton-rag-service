"""Two baseline retrievers over the real services. Both return k DISTINCT,
globally-ranked product source_ids via rank_products."""

from app.services.conversation_state import ConversationState
from eval.rank import rank_products

OVER_FETCH_FACTOR = 8


class NaiveVectorRetriever:
    """Ablation control: raw-query vector search, product docs only."""

    def __init__(self, embedder, retrieval):
        self._embedder = embedder
        self._retrieval = retrieval

    async def retrieve(self, query: str, k: int) -> list[str]:
        emb = await self._embedder.embed(query)
        chunks = await self._retrieval.search(
            emb, k=k * OVER_FETCH_FACTOR, doc_type="product"
        )
        return rank_products(chunks, k)


class ProductionRetriever:
    """Mirrors app/routers/chat.py: contextualized retrieval_query + category
    filter, no doc_type filter (static docs compete, as in production)."""

    def __init__(self, embedder, retrieval, qu):
        self._embedder = embedder
        self._retrieval = retrieval
        self._qu = qu

    async def retrieve(self, query: str, k: int) -> list[str]:
        understanding = await self._qu.analyze(query, ConversationState())
        emb = await self._embedder.embed(understanding.retrieval_query)
        categories = understanding.categories or None
        chunks = await self._retrieval.search(
            emb, k=k * OVER_FETCH_FACTOR, categories=categories
        )
        return rank_products(chunks, k)
