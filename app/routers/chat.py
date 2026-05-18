"""POST /chat — RAG chat endpoint (Phase 3: retrieval + LLM, no tool calling)."""

from fastapi import APIRouter, HTTPException, Request

from app.core.config import settings
from app.core.prompts import SYSTEM_PROMPT
from app.models.schemas import ChatRequest, ChatResponse, SourceRef
from app.services.embedding import EmbeddingService
from app.services.llm import LLMService
from app.services.retrieval import Chunk, RetrievalService

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest, http_request: Request) -> ChatResponse:
    # Inject services từ app.state (set ở lifespan của main.py)
    embedding_svc: EmbeddingService = http_request.app.state.embedding
    retrieval_svc: RetrievalService = http_request.app.state.retrieval
    llm_svc: LLMService = http_request.app.state.llm

    query = request.message.strip()
    if len(query) < settings.min_query_length:
        raise HTTPException(status_code=400, detail="Câu hỏi quá ngắn")

    # 1. Embed user query
    query_vec = await embedding_svc.embed(query)

    # 2. Retrieve top-k chunks
    chunks = await retrieval_svc.search(query_vec)

    # 3. Build context block
    context = _format_context(chunks)

    # 4. Build messages: system + history + user
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context)}
    ]
    for m in request.chat_history:
        messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": query})

    # 5. Call LLM
    answer = await llm_svc.chat(messages)

    # 6. Build source refs (unique by doc_type+source_id)
    seen: set[tuple[str, str]] = set()
    sources: list[SourceRef] = []
    for c in chunks:
        key = (c.doc_type, c.source_id)
        if key in seen:
            continue
        seen.add(key)
        sources.append(SourceRef(doc_type=c.doc_type, source_id=c.source_id))

    return ChatResponse(answer=answer, sources=sources)


def _format_context(chunks: list[Chunk]) -> str:
    """Format chunks thành block dễ đọc cho LLM."""
    if not chunks:
        return "(Không tìm thấy thông tin liên quan trong cơ sở dữ liệu.)"
    parts = []
    for c in chunks:
        parts.append(f"[{c.doc_type} / {c.source_id}]\n{c.content}")
    return "\n\n---\n\n".join(parts)
