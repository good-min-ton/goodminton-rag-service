import httpx
import pytest

from app.services.embed_client import EmbedClient, EmbedUnavailable


@pytest.mark.asyncio
async def test_embed_image_returns_vector_and_posts_file_field():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["content_type"] = request.headers.get("content-type", "")
        captured["body"] = request.content
        return httpx.Response(200, json={"embedding": [0.1] * 768})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        vec = await EmbedClient(client).embed_image(b"rawbytes")

    assert len(vec) == 768
    assert captured["content_type"].startswith("multipart/form-data")
    assert b'name="file"' in captured["body"]  # H2: multipart field 'file'


@pytest.mark.asyncio
async def test_embed_image_non_200_raises_embed_unavailable():
    transport = httpx.MockTransport(lambda req: httpx.Response(500, text="boom"))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(EmbedUnavailable):
            await EmbedClient(client).embed_image(b"x")


@pytest.mark.asyncio
async def test_embed_image_connect_error_raises_embed_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(EmbedUnavailable):
            await EmbedClient(client).embed_image(b"x")


@pytest.mark.asyncio
async def test_embed_image_declares_image_content_type():
    # Regression (live E2E): the embed-service rejects a non-image Content-Type
    # with HTTP 400, so the multipart file part must declare an image/* type —
    # NOT application/octet-stream, which broke both backfill and /search/image.
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(200, json={"embedding": [0.1] * 768})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        await EmbedClient(client).embed_image(b"rawbytes")

    assert b"Content-Type: image/" in captured["body"]
    assert b"application/octet-stream" not in captured["body"]
