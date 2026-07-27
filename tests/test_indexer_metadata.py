from unittest.mock import AsyncMock
from app.services.indexer import ProductIndexer


async def test_index_product_writes_category_metadata(truncate_kb):
    pool = truncate_kb
    embedding = AsyncMock()
    embedding.embed.return_value = [0.0] * 1024
    client = AsyncMock()
    client.get_for_rag.return_value = {
        "name": "Quần A",
        "brand": "X",
        "category": "Quần cầu lông",
        "specifications": [],
        "description": "mô tả",
    }
    indexer = ProductIndexer(pool, embedding, client)
    await indexer.index_product(101)
    async with pool.acquire() as conn:
        cat = await conn.fetchval(
            "SELECT metadata->>'category' FROM kb_chunks "
            "WHERE doc_type='product' AND source_id='101' LIMIT 1"
        )
    assert cat == "Quần cầu lông"
