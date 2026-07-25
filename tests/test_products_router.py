from unittest.mock import AsyncMock

import httpx
import pytest


def _make_client_with_desc(desc_service):
    from app.main import app

    app.state.description = desc_service
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_post_description_returns_200_and_body():
    desc = AsyncMock()
    desc.generate.return_value = (
        "Mô tả sản phẩm tuyệt vời.",
        "qwen2.5:14b-instruct-q4_K_M",
    )
    async with _make_client_with_desc(desc) as client:
        resp = await client.post("/products/123/description", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert body["product_id"] == 123
    assert body["description"] == "Mô tả sản phẩm tuyệt vời."
    assert body["style"] == "ban_hang"
    assert body["length"] == "medium"
    # Pin the exact call: arg order (style/length are both str at the same
    # call site) + keywords passthrough. An empty {} body uses the schema
    # defaults, so generate() must be awaited with these literals.
    desc.generate.assert_awaited_once_with(123, "ban_hang", "medium", [])


@pytest.mark.asyncio
async def test_post_description_invalid_style_returns_422():
    desc = AsyncMock()
    async with _make_client_with_desc(desc) as client:
        resp = await client.post(
            "/products/123/description", json={"style": "clickbait"}
        )
    assert resp.status_code == 422
    desc.generate.assert_not_called()


@pytest.mark.asyncio
async def test_post_description_product_not_found_maps_404():
    desc = AsyncMock()
    request = httpx.Request("GET", "http://x")
    response = httpx.Response(404, request=request)
    desc.generate.side_effect = httpx.HTTPStatusError(
        "404", request=request, response=response
    )
    async with _make_client_with_desc(desc) as client:
        resp = await client.post("/products/999/description", json={})
    assert resp.status_code == 404
    assert "sản phẩm" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_post_description_upstream_llm_error_maps_502():
    desc = AsyncMock()
    request = httpx.Request("POST", "http://ollama")
    response = httpx.Response(500, request=request)
    desc.generate.side_effect = httpx.HTTPStatusError(
        "500", request=request, response=response
    )
    async with _make_client_with_desc(desc) as client:
        resp = await client.post("/products/123/description", json={})
    assert resp.status_code == 502
