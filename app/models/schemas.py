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


class BranchStock(BaseModel):
    """Stock at a non-central store. Walk-in only: an ONLINE order cannot draw on
    it, so this is advice ("it is in stock at Q7"), never a basis for ordering."""

    store_id: int | None = None
    store_name: str | None = None
    quantity: int


class OrderOption(BaseModel):
    """One orderable variant, priced and stock-checked, ready to be a chip in the
    frontend's picker. `unit_price` already applies sale-price precedence so the
    UI never has to repeat that rule."""

    variant_id: str
    size: str | None = None
    color: str | None = None
    unit_price: float
    # Central-store quantity: the picker caps the quantity stepper at this, so a
    # customer cannot assemble an order the checkout would reject.
    orderable: int
    branches: list[BranchStock] = Field(default_factory=list)


class OrderSelection(BaseModel):
    """Everything the frontend needs to let a customer pick a variant without
    another LLM turn.

    The model used to ask for size and colour in prose and then map the reply
    back to a variant_id itself, which is both slow (a generation per question)
    and unreliable. Here it names the product once; picking is deterministic.

    Options are a list, not a size x colour matrix: `UNIQUE (product_id,
    color_id, size_id)` does not guarantee every combination exists, and both
    columns are nullable, so the frontend derives the chips it can offer from
    the variants that actually exist.
    """

    product_id: str
    product_name: str
    currency: str = "VND"
    options: list[OrderOption] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    # Legacy prose-scraped ids; kept for back-compat. Cards now use display_products.
    products: list[str] = Field(default_factory=list)
    # Set when the customer wants to buy. The frontend renders it as a picker and
    # builds the priced draft itself once they choose, so the backend has no draft
    # of its own to send.
    order_selection: OrderSelection | None = None
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
