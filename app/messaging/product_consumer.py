"""RabbitMQ consumer for product change events."""

import json
import logging

import aio_pika
from aio_pika.abc import AbstractIncomingMessage

from app.core.config import settings
from app.services.image_indexer import ImageIndexer
from app.services.indexer import ProductIndexer

log = logging.getLogger(__name__)

SEMANTIC_FIELDS = {"name", "description", "specs", "brand", "category"}


class ProductConsumer:
    def __init__(self, indexer: ProductIndexer, image_indexer: ImageIndexer):
        self._indexer = indexer
        self._image_indexer = image_indexer
        self._connection: aio_pika.abc.AbstractRobustConnection | None = None

    async def start(self) -> None:
        self._connection = await aio_pika.connect_robust(settings.resolved_rabbitmq_url)
        channel = await self._connection.channel()
        await channel.set_qos(prefetch_count=5)

        queue = await channel.declare_queue(
            settings.rag_product_queue,
            durable=True,
            arguments={
                "x-dead-letter-exchange": "",
                "x-dead-letter-routing-key": settings.rag_product_dlq,
            },
        )

        await queue.consume(self._on_message)
        log.info("Consumer listening on queue %s", settings.rag_product_queue)

    async def stop(self) -> None:
        if self._connection:
            await self._connection.close()

    async def _on_message(self, message: AbstractIncomingMessage) -> None:
        # requeue=False: failure routes message to DLQ instead of infinite retry
        async with message.process(requeue=False):
            try:
                event = json.loads(message.body)
                await self._handle(event)
            except Exception:
                log.exception("Failed to handle message: %s", message.body[:200])
                raise

    async def _handle(self, event: dict) -> None:
        action = event.get("action")
        product_id = event.get("productId")
        fields = set(event.get("fieldsChanged") or [])

        if not product_id:
            log.warning("Event missing productId: %s", event)
            return

        if action == "deleted":
            await self._indexer.delete_product(product_id)
            await self._image_indexer.delete_product_images(product_id)
            return

        if action in ("created", "updated"):
            if fields & SEMANTIC_FIELDS:
                await self._indexer.index_product(product_id)
            if "images" in fields:
                await self._image_indexer.index_product_images(product_id)
            if not (fields & SEMANTIC_FIELDS) and "images" not in fields:
                log.debug(
                    "Skip product %s — no indexed fields changed (%s)",
                    product_id,
                    fields,
                )
            return

        log.warning("Unknown action '%s' for product %s", action, product_id)
