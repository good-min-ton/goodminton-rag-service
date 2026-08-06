from unittest.mock import AsyncMock

import httpx
import pytest

from app.main import app
from app.services.embed_client import EmbedUnavailable


def _client() -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_search_text_returns_product_ids():
    from app.routers import search as search_mod

    search_mod._text_hits.clear()
    app.state.embed = AsyncMock()
    app.state.embed.embed_text.return_value = [0.1] * 768
    app.state.image_search = AsyncMock()
    app.state.image_search.search.return_value = ["42", "17"]

    async with _client() as c:
        r = await c.post("/search/text", json={"query": "vợt màu đỏ"})

    assert r.status_code == 200
    assert r.json() == {"product_ids": ["42", "17"]}  # H1


@pytest.mark.asyncio
async def test_search_text_empty_query_returns_400():
    from app.routers import search as search_mod

    search_mod._text_hits.clear()
    app.state.embed = AsyncMock()
    app.state.image_search = AsyncMock()

    async with _client() as c:
        r = await c.post("/search/text", json={"query": "   "})

    assert r.status_code == 400
    app.state.embed.embed_text.assert_not_called()


@pytest.mark.asyncio
async def test_search_text_over_length_query_returns_400():
    from app.routers import search as search_mod

    search_mod._text_hits.clear()
    app.state.embed = AsyncMock()
    app.state.image_search = AsyncMock()

    async with _client() as c:
        r = await c.post("/search/text", json={"query": "a" * 401})

    assert r.status_code == 400
    app.state.embed.embed_text.assert_not_called()


@pytest.mark.asyncio
async def test_search_text_embed_unavailable_returns_503():
    from app.routers import search as search_mod

    search_mod._text_hits.clear()
    app.state.embed = AsyncMock()
    app.state.embed.embed_text.side_effect = EmbedUnavailable("down")
    app.state.image_search = AsyncMock()

    async with _client() as c:
        r = await c.post("/search/text", json={"query": "vợt"})

    assert r.status_code == 503  # H7


@pytest.mark.asyncio
async def test_search_text_rate_limited_returns_429(monkeypatch):
    from app.routers import search as search_mod

    search_mod._text_hits.clear()
    monkeypatch.setattr(search_mod, "_TEXT_RATE_MAX", 1)
    app.state.embed = AsyncMock()
    app.state.embed.embed_text.return_value = [0.1] * 768
    app.state.image_search = AsyncMock()
    app.state.image_search.search.return_value = []

    async with _client() as c:
        r1 = await c.post("/search/text", json={"query": "vợt"})
        r2 = await c.post("/search/text", json={"query": "vợt"})

    assert r1.status_code == 200
    assert r2.status_code == 429  # H5 — own bucket
    search_mod._text_hits.clear()
