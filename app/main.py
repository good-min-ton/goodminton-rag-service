"""FastAPI entrypoint for rag-service."""

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.db import create_pool
from app.core.tracing import langfuse
from app.messaging.product_consumer import ProductConsumer
from app.models.schemas import HealthResponse
from app.routers import chat as chat_router
from app.routers import products as products_router
from app.services.description import DescriptionService
from app.services.embedding import EmbeddingService
from app.services.indexer import ProductIndexer
from app.services.llm import LLMService
from app.services.product_client import ProductClient
from app.services.retrieval import RetrievalService
from app.services.similar import SimilarProductsService
from app.services.tools import ToolDispatcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = await create_pool()
    http_client = httpx.AsyncClient()

    import redis.asyncio as aioredis  # local import keeps module import cheap

    redis_client = (
        aioredis.from_url(settings.redis_url, decode_responses=True)
        if settings.redis_url
        else None
    )

    embedding = EmbeddingService(http_client)
    product_client = ProductClient(http_client)
    indexer = ProductIndexer(pool, embedding, product_client)
    consumer = ProductConsumer(indexer)
    similar_svc = SimilarProductsService(pool)
    tool_dispatcher = ToolDispatcher(product_client, similar_svc)

    app.state.pool = pool
    app.state.http = http_client
    app.state.embedding = embedding
    app.state.retrieval = RetrievalService(pool)
    app.state.llm = LLMService(http_client)

    from app.services.query_understanding import QueryUnderstandingService

    app.state.query_understanding = QueryUnderstandingService(app.state.llm)
    app.state.indexer = indexer
    app.state.consumer = consumer
    app.state.tool_dispatcher = tool_dispatcher
    app.state.similar = similar_svc
    app.state.description = DescriptionService(
        llm=app.state.llm, product_client=product_client
    )

    from app.services.conversation_state import ConversationStateStore

    app.state.redis = redis_client
    app.state.conversation_state = ConversationStateStore(redis_client).with_ttl(
        settings.chat_state_ttl_seconds
    )

    await consumer.start()

    yield

    await consumer.stop()
    if redis_client is not None:
        await redis_client.aclose()
    await http_client.aclose()
    await pool.close()
    langfuse.flush()


app = FastAPI(
    title="Goodminton RAG Service",
    version="0.1.0",
    description="RAG chatbot tư vấn sản phẩm cầu lông.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router.router)
app.include_router(products_router.router)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")
