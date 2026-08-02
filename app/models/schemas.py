from typing import Literal

from pydantic import BaseModel, Field

from app.services.conversation_state import ConversationState


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    chat_history: list[ChatMessage] = Field(default_factory=list, max_length=20)
    session_id: str | None = None
    # Set by the frontend on the turn(s) after it places an order, so the backend
    # can advance the order state machine to ORDER_CONFIRMED. Backend never places
    # an order — this is a read-only signal.
    order_placed_id: int | None = None


class SourceRef(BaseModel):
    doc_type: Literal["static", "product"]
    source_id: str


class OrderDraftItem(BaseModel):
    product_id: str
    variant_id: str
    product_name: str
    size: str | None = None
    color: str | None = None
    quantity: int
    unit_price: float
    line_total: float
    in_stock: bool


class OrderDraft(BaseModel):
    items: list[OrderDraftItem] = Field(default_factory=list)
    total: float = 0.0
    currency: str = "VND"
    warnings: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    # Legacy prose-scraped ids; kept for back-compat. Cards now use display_products.
    products: list[str] = Field(default_factory=list)
    order_draft: OrderDraft | None = None
    # Foundation additions:
    intent: str | None = None
    categories: list[str] = Field(default_factory=list)
    # The ONLY ids the frontend renders as cards for this message (structured, not scraped).
    display_products: list[int] = Field(default_factory=list)
    conversation_state: ConversationState = Field(default_factory=ConversationState)


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


class FeedbackRequest(BaseModel):
    session_id: str | None = None
    helpful: bool
    comment: str | None = Field(default=None, max_length=1000)
