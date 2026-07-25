from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    chat_history: list[ChatMessage] = Field(default_factory=list, max_length=20)
    session_id: str | None = None


class SourceRef(BaseModel):
    doc_type: Literal["static", "product"]
    source_id: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceRef]


class HealthResponse(BaseModel):
    status: str


class SimilarProduct(BaseModel):
    product_id: str
    name: str | None = None
    similarity: float
    distance: float
    chunk_count: int


class SimilarProductsResponse(BaseModel):
    product_id: str
    count: int
    results: list[SimilarProduct]


class DescriptionRequest(BaseModel):
    style: Literal["ban_hang", "chuyen_nghiep", "than_thien", "seo"] = "ban_hang"
    length: Literal["short", "medium", "long"] = "medium"
    keywords: list[str] = Field(default_factory=list, max_length=10)


class DescriptionResponse(BaseModel):
    product_id: int
    description: str
    model: str
    style: str
    length: str
