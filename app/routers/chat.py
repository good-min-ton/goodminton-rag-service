"""POST /chat — RAG chat endpoint with retrieval + LLM + tool calling."""

import logging

from fastapi import APIRouter, HTTPException, Request

from app.core.config import settings
from app.core.prompts import SYSTEM_PROMPT
from app.models.schemas import ChatRequest, ChatResponse, SourceRef
from app.services.embedding import EmbeddingService
from app.services.llm import LLMService
from app.services.retrieval import Chunk, RetrievalService
from app.services.tools import TOOL_SCHEMAS, ToolDispatcher

log = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

MAX_TOOL_ITERATIONS = 10


@router.post("")
async def chat(request: ChatRequest, http_request: Request) -> ChatResponse:
    embedding_svc: EmbeddingService = http_request.app.state.embedding
    retrieval_svc: RetrievalService = http_request.app.state.retrieval
    llm_svc: LLMService = http_request.app.state.llm
    tool_dispatcher: ToolDispatcher = http_request.app.state.tool_dispatcher

    query = request.message.strip()
    if len(query) < settings.min_query_length:
        raise HTTPException(status_code=400, detail="Câu hỏi quá ngắn")

    query_vec = await embedding_svc.embed(query)
    chunks = await retrieval_svc.search(query_vec)
    context = _format_context(chunks)
    product_ids = _extract_product_ids(chunks)

    system_content = SYSTEM_PROMPT.format(context=context)
    if product_ids:
        system_content += (
            "\n\nDanh sách product_id để gọi tool (KHÔNG nhắc ID trong câu trả lời):\n"
            + ", ".join(product_ids)
        )

    messages: list[dict] = [{"role": "system", "content": system_content}]
    for m in request.chat_history:
        messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": query})

    answer = await _run_tool_loop(llm_svc, tool_dispatcher, messages)

    return ChatResponse(answer=answer, sources=_unique_sources(chunks))


async def _run_tool_loop(
    llm: LLMService, dispatcher: ToolDispatcher, messages: list[dict]
) -> str:
    """Loop: LLM may call tools; execute, feed back, repeat until text answer."""
    for iteration in range(MAX_TOOL_ITERATIONS):
        msg = await llm.chat_with_tools(messages, TOOL_SCHEMAS)
        tool_calls = msg.get("tool_calls") or []
        log.info(
            "iter=%d tool_calls=%d content_preview=%r",
            iteration,
            len(tool_calls),
            (msg.get("content") or "")[:100],
        )

        if not tool_calls:
            return msg.get("content") or ""

        messages.append(msg)

        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            arguments = fn.get("arguments") or {}
            log.info("Tool call: %s(%s)", name, arguments)
            result = await dispatcher.execute(name, arguments)
            messages.append({"role": "tool", "name": name, "content": result})

    log.warning("Tool loop hit max iterations (%d)", MAX_TOOL_ITERATIONS)
    return "Xin lỗi, mình không xử lý được yêu cầu này. Vui lòng liên hệ shop."


def _unique_sources(chunks: list[Chunk]) -> list[SourceRef]:
    seen: set[tuple[str, str]] = set()
    out: list[SourceRef] = []
    for c in chunks:
        key = (c.doc_type, c.source_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(SourceRef(doc_type=c.doc_type, source_id=c.source_id))
    return out


def _format_context(chunks: list[Chunk]) -> str:
    if not chunks:
        return "(Không tìm thấy thông tin liên quan trong cơ sở dữ liệu.)"
    return "\n\n---\n\n".join(c.content for c in chunks)


def _extract_product_ids(chunks: list[Chunk]) -> list[str]:
    """Unique product source_ids in retrieval order — for tool-calling hints in prompt."""
    seen: set[str] = set()
    out: list[str] = []
    for c in chunks:
        if c.doc_type != "product":
            continue
        if c.source_id in seen:
            continue
        seen.add(c.source_id)
        out.append(c.source_id)
    return out
