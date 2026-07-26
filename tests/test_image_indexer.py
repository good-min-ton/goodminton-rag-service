# tests/test_image_indexer.py
from unittest.mock import AsyncMock

import httpx
import pytest

from app.services.embed_client import EmbedUnavailable
from app.services.image_indexer import ImageIndexer


def _http_client() -> httpx.AsyncClient:
    # Every image URL "downloads" to the same small byte payload.
    return httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(200, content=b"imgbytes")
        )
    )


@pytest.mark.asyncio
async def test_one_image_fails_others_still_indexed(
    truncate_image_embeddings, make_embedding_768
):
    pool = truncate_image_embeddings
    pc = AsyncMock()
    pc.get_product_images.return_value = [
        {"resourceId": 10, "url": "http://cdn/a.jpg", "sortOrder": 0},
        {"resourceId": 11, "url": "http://cdn/b.jpg", "sortOrder": 1},
    ]
    embed = AsyncMock()
    # First image fails to embed, second succeeds.
    embed.embed_image.side_effect = [
        EmbedUnavailable("bad"),
        make_embedding_768({0: 1.0}),
    ]

    async with _http_client() as http:
        indexer = ImageIndexer(pool, embed, pc, http)
        n = await indexer.index_product_images(5)

    assert n == 1  # only the second image embedded (H9 resilient)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT resource_id FROM product_image_embeddings WHERE product_id = 5"
        )
    assert [r["resource_id"] for r in rows] == [11]


@pytest.mark.asyncio
async def test_all_images_fail_does_not_wipe_existing(
    truncate_image_embeddings, make_embedding_768
):
    from tests.conftest import seed_image_embedding

    pool = truncate_image_embeddings
    # Pre-existing indexed row for product 5.
    await seed_image_embedding(
        pool, 99, 5, "http://cdn/old.jpg", make_embedding_768({0: 1.0})
    )

    pc = AsyncMock()
    pc.get_product_images.return_value = [
        {"resourceId": 10, "url": "http://cdn/a.jpg", "sortOrder": 0},
    ]
    embed = AsyncMock()
    embed.embed_image.side_effect = EmbedUnavailable("bad")  # ALL fail

    async with _http_client() as http:
        indexer = ImageIndexer(pool, embed, pc, http)
        n = await indexer.index_product_images(5)

    assert n == 0
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT resource_id FROM product_image_embeddings WHERE product_id = 5"
        )
    assert [r["resource_id"] for r in rows] == [99]  # existing row NOT wiped (H9)


@pytest.mark.asyncio
async def test_delete_product_images_removes_rows(
    truncate_image_embeddings, make_embedding_768
):
    from tests.conftest import seed_image_embedding

    pool = truncate_image_embeddings
    await seed_image_embedding(
        pool, 99, 5, "http://cdn/x.jpg", make_embedding_768({0: 1.0})
    )

    async with _http_client() as http:
        indexer = ImageIndexer(pool, AsyncMock(), AsyncMock(), http)
        deleted = await indexer.delete_product_images(5)

    assert deleted == 1
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT resource_id FROM product_image_embeddings WHERE product_id = 5"
        )
    assert rows == []
