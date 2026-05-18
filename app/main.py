"""FastAPI entrypoint cho rag-service.

Phase 3: chat endpoint với retrieval + LLM, chưa có tool calling.
"""

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.db import create_pool
from app.models.schemas import HealthResponse
from app.routers import chat as chat_router
from app.services.embedding import EmbeddingService
from app.services.llm import LLMService
from app.services.retrieval import RetrievalService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Khởi tạo resources: DB pool + HTTP client + service singletons."""
    pool = await create_pool()
    http_client = httpx.AsyncClient()

    app.state.pool = pool
    app.state.http = http_client
    app.state.embedding = EmbeddingService(http_client)
    app.state.retrieval = RetrievalService(pool)
    app.state.llm = LLMService(http_client)

    yield

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
