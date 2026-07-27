from unittest.mock import AsyncMock

import httpx
import pytest

from app.main import app
from app.services.embed_client import EmbedUnavailable


def _client() -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_search_image_returns_product_ids():
    app.state.embed = AsyncMock()
    app.state.embed.embed_image.return_value = [0.1] * 768
    app.state.image_search = AsyncMock()
    app.state.image_search.search.return_value = ["42", "17"]

    async with _client() as c:
        r = await c.post(
            "/search/image", files={"file": ("q.jpg", b"imgbytes", "image/jpeg")}
        )

    assert r.status_code == 200
    assert r.json() == {"product_ids": ["42", "17"]}  # H1


@pytest.mark.asyncio
async def test_search_image_embed_unavailable_returns_503():
    app.state.embed = AsyncMock()
    app.state.embed.embed_image.side_effect = EmbedUnavailable("down")
    app.state.image_search = AsyncMock()

    async with _client() as c:
        r = await c.post(
            "/search/image", files={"file": ("q.jpg", b"imgbytes", "image/jpeg")}
        )

    assert r.status_code == 503  # H7


@pytest.mark.asyncio
async def test_search_image_rejects_non_image_content_type():
    app.state.embed = AsyncMock()
    app.state.image_search = AsyncMock()

    async with _client() as c:
        r = await c.post(
            "/search/image", files={"file": ("q.txt", b"hello", "text/plain")}
        )

    assert r.status_code == 400


@pytest.mark.asyncio
async def test_search_image_rate_limited_returns_429(monkeypatch):
    from app.routers import search as search_mod

    search_mod._hits.clear()
    monkeypatch.setattr(search_mod, "_RATE_MAX", 1)
    app.state.embed = AsyncMock()
    app.state.embed.embed_image.return_value = [0.1] * 768
    app.state.image_search = AsyncMock()
    app.state.image_search.search.return_value = []

    async with _client() as c:
        r1 = await c.post(
            "/search/image", files={"file": ("q.jpg", b"imgbytes", "image/jpeg")}
        )
        r2 = await c.post(
            "/search/image", files={"file": ("q.jpg", b"imgbytes", "image/jpeg")}
        )

    assert r1.status_code == 200
    assert r2.status_code == 429  # H5
    search_mod._hits.clear()
