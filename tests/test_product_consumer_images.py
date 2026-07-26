# tests/test_product_consumer_images.py
from unittest.mock import AsyncMock

import pytest

from app.messaging.product_consumer import ProductConsumer


@pytest.mark.asyncio
async def test_images_field_triggers_image_reindex_only():
    idx = AsyncMock()
    img = AsyncMock()
    consumer = ProductConsumer(idx, img)

    await consumer._handle(
        {"action": "updated", "productId": 5, "fieldsChanged": ["images"]}
    )

    img.index_product_images.assert_awaited_once_with(5)
    idx.index_product.assert_not_called()  # 'images' is not a SEMANTIC_FIELD


@pytest.mark.asyncio
async def test_semantic_and_images_triggers_both():
    idx = AsyncMock()
    img = AsyncMock()
    consumer = ProductConsumer(idx, img)

    await consumer._handle(
        {"action": "updated", "productId": 5, "fieldsChanged": ["name", "images"]}
    )

    idx.index_product.assert_awaited_once_with(5)
    img.index_product_images.assert_awaited_once_with(5)


@pytest.mark.asyncio
async def test_semantic_only_event_does_not_reindex_images():
    idx = AsyncMock()
    img = AsyncMock()
    consumer = ProductConsumer(idx, img)

    await consumer._handle(
        {"action": "updated", "productId": 5, "fieldsChanged": ["name"]}
    )

    idx.index_product.assert_awaited_once_with(5)
    img.index_product_images.assert_not_called()


@pytest.mark.asyncio
async def test_deleted_removes_text_and_image_rows():
    idx = AsyncMock()
    img = AsyncMock()
    consumer = ProductConsumer(idx, img)

    await consumer._handle({"action": "deleted", "productId": 5})

    idx.delete_product.assert_awaited_once_with(5)
    img.delete_product_images.assert_awaited_once_with(5)
