"""FastAPI entrypoint for rag-service."""

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

from app.core.config import settings
from app.core.db import create_pool
from app.messaging.product_consumer import ProductConsumer
from app.models.schemas import HealthResponse
from app.routers import chat as chat_router
from app.services.embedding import EmbeddingService
from app.services.indexer import ProductIndexer
from app.services.llm import LLMService
from app.services.product_client import ProductClient
from app.services.retrieval import RetrievalService
from app.services.tools import ToolDispatcher


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = await create_pool()
    http_client = httpx.AsyncClient()

    embedding = EmbeddingService(http_client)
    product_client = ProductClient(http_client)
    indexer = ProductIndexer(pool, embedding, product_client)
    consumer = ProductConsumer(indexer)
    tool_dispatcher = ToolDispatcher(product_client)

    app.state.pool = pool
    app.state.http = http_client
    app.state.embedding = embedding
    app.state.retrieval = RetrievalService(pool)
    app.state.llm = LLMService(http_client)
    app.state.indexer = indexer
    app.state.consumer = consumer
    app.state.tool_dispatcher = tool_dispatcher

    await consumer.start()

    yield

    await consumer.stop()
    await http_client.aclose()
    await pool.close()


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
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(chat_router.router)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")
