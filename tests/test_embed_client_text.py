import json

import httpx
import pytest

from app.services.embed_client import EmbedClient, EmbedUnavailable


def _c(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_embed_text_returns_vector():
    async def h(req):
        assert req.url.path == "/embed/text"
        assert json.loads(req.content.decode())["text"] == "vợt đỏ"
        return httpx.Response(200, json={"embedding": [0.1] * 768})

    async with _c(h) as c:
        v = await EmbedClient(c).embed_text("vợt đỏ")
    assert len(v) == 768


@pytest.mark.asyncio
async def test_embed_text_raises_on_non_200():
    async def h(req):
        return httpx.Response(500)

    async with _c(h) as c:
        with pytest.raises(EmbedUnavailable):
            await EmbedClient(c).embed_text("x")
