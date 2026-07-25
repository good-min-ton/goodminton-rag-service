# tests/test_tools_similar.py
import json
from unittest.mock import AsyncMock

import pytest

from app.services.similar import ProductNotIndexedError, SimilarResult
from app.services.tools import TOOL_SCHEMAS, ToolDispatcher


def test_recommend_similar_products_schema_present():
    names = {t["function"]["name"] for t in TOOL_SCHEMAS}
    assert "recommend_similar_products" in names


@pytest.mark.asyncio
async def test_recommend_similar_products_dispatch_maps_similarity():
    similar = AsyncMock()
    similar.find_similar.return_value = [
        SimilarResult(product_id="2", name="B", distance=0.1, chunk_count=1)
    ]
    dispatcher = ToolDispatcher(product_client=AsyncMock(), similar=similar)
    out = await dispatcher.execute("recommend_similar_products", {"product_id": 1})
    parsed = json.loads(out)
    assert parsed[0]["product_id"] == "2"
    assert parsed[0]["similarity"] == pytest.approx(0.9)  # 1.0 - distance
    assert parsed[0]["distance"] == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_recommend_similar_products_no_peers_returns_empty():
    similar = AsyncMock()
    similar.find_similar.return_value = []
    dispatcher = ToolDispatcher(product_client=AsyncMock(), similar=similar)
    out = await dispatcher.execute("recommend_similar_products", {"product_id": 1})
    assert json.loads(out) == []


@pytest.mark.asyncio
async def test_recommend_similar_products_not_indexed_returns_error():
    similar = AsyncMock()
    similar.find_similar.side_effect = ProductNotIndexedError("999")
    dispatcher = ToolDispatcher(product_client=AsyncMock(), similar=similar)
    out = await dispatcher.execute("recommend_similar_products", {"product_id": 999})
    parsed = json.loads(out)
    assert "error" in parsed
