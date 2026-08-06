import httpx
import pytest
from eval.rerank_client import RerankClient, RerankUnavailable


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_rerank_returns_scores():
    async def handler(req):
        assert req.url.path == "/rerank"
        return httpx.Response(200, json={"scores": [0.9, 0.1]})

    async with _client(handler) as c:
        scores = await RerankClient(c).rerank("q", ["a", "b"])
    assert scores == [0.9, 0.1]


@pytest.mark.asyncio
async def test_rerank_raises_on_error_status():
    async def handler(req):
        return httpx.Response(500)

    async with _client(handler) as c:
        with pytest.raises(RerankUnavailable):
            await RerankClient(c).rerank("q", ["a"])


@pytest.mark.asyncio
async def test_rerank_raises_on_missing_scores():
    async def handler(req):
        return httpx.Response(200, json={})

    async with _client(handler) as c:
        with pytest.raises(RerankUnavailable):
            await RerankClient(c).rerank("q", ["a", "b"])


@pytest.mark.asyncio
async def test_rerank_raises_on_length_mismatch():
    async def handler(req):
        return httpx.Response(200, json={"scores": [0.1]})

    async with _client(handler) as c:
        with pytest.raises(RerankUnavailable):
            await RerankClient(c).rerank("q", ["a", "b"])
