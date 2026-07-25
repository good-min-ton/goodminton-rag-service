"""POST /chat — RAG chat endpoint with retrieval + LLM + tool calling."""

import json
import logging

from fastapi import APIRouter, HTTPException, Request
from langfuse import propagate_attributes

from app.core.config import settings
from app.core.prompts import SYSTEM_PROMPT
from app.core.tracing import langfuse
from app.models.schemas import ChatRequest, ChatResponse, SourceRef
from app.services.embedding import EmbeddingService
from app.services.llm import LLMService
from app.services.retrieval import Chunk, RetrievalService
from app.services.tools import TOOL_SCHEMAS, ToolDispatcher

log = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

MAX_TOOL_ITERATIONS = 10
MAX_REPEATED_CALLS = 2


@router.post("")
async def chat(request: ChatRequest, http_request: Request) -> ChatResponse:
    embedding_svc: EmbeddingService = http_request.app.state.embedding
    retrieval_svc: RetrievalService = http_request.app.state.retrieval
    llm_svc: LLMService = http_request.app.state.llm
    tool_dispatcher: ToolDispatcher = http_request.app.state.tool_dispatcher

    query = request.message.strip()
    if len(query) < settings.min_query_length:
        raise HTTPException(status_code=400, detail="Câu hỏi quá ngắn")

    with (
        propagate_attributes(session_id=request.session_id or None, trace_name="chat"),
        langfuse.start_as_current_observation(
            name="chat", as_type="span", input=query
        ) as root,
    ):
        with langfuse.start_as_current_observation(
            name="embed",
            as_type="embedding",
            input=query,
            model=settings.embedding_model,
        ):
            query_vec = await embedding_svc.embed(query)

        with langfuse.start_as_current_observation(
            name="retrieval", as_type="retriever", input=query
        ) as rspan:
            chunks = await retrieval_svc.search(query_vec)
            rspan.update(
                output=[
                    {"doc_type": c.doc_type, "source_id": c.source_id} for c in chunks
                ]
            )

        context = _format_context(chunks)
        product_ids = _extract_product_ids(chunks)

        system_content = SYSTEM_PROMPT.format(context=context)
        if product_ids:
            system_content += (
                "\n\nDanh sách product_id hợp lệ để gọi tool (KHÔNG nhắc ID trong câu "
                "trả lời, KHÔNG dùng ID ngoài danh sách này):\n"
                + ", ".join(product_ids)
            )
        else:
            system_content += (
                "\n\nNgữ cảnh không chứa sản phẩm nào. KHÔNG gọi tool với ID tự nghĩ "
                "ra; nếu khách hỏi giá hoặc tồn kho, trả lời rằng bạn không tìm thấy "
                "sản phẩm phù hợp."
            )

        messages: list[dict] = [{"role": "system", "content": system_content}]
        for m in request.chat_history:
            messages.append({"role": m.role, "content": m.content})
        messages.append({"role": "user", "content": query})

        answer = await _run_tool_loop(llm_svc, tool_dispatcher, messages)

        root.update(output=answer)
        return ChatResponse(answer=answer, sources=_unique_sources(chunks))


async def _run_tool_loop(
    llm: LLMService, dispatcher: ToolDispatcher, messages: list[dict]
) -> str:
    """Loop: LLM may call tools; execute, feed back, repeat until text answer.

    Identical calls are cached; after MAX_REPEATED_CALLS repeats the model is
    clearly stuck (e.g. retrying an invalid ID), so we force a final answer
    without tools instead of burning the remaining iterations.
    """
    executed: dict[tuple[str, str], str] = {}
    repeats = 0
    for iteration in range(MAX_TOOL_ITERATIONS):
        with langfuse.start_as_current_observation(
            name="llm.chat_with_tools",
            as_type="generation",
            input=messages,
            model=settings.llm_model,
        ) as gen:
            msg = await llm.chat_with_tools(messages, TOOL_SCHEMAS)
            gen.update(output=msg)
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
            key = (name, json.dumps(arguments, sort_keys=True, default=str))
            if key in executed:
                repeats += 1
                log.info("Repeated tool call %s(%s), cached result", name, arguments)
                result = executed[key]
            else:
                log.info("Tool call: %s(%s)", name, arguments)
                with langfuse.start_as_current_observation(
                    name=f"tool.{name}", as_type="tool", input=arguments
                ) as tspan:
                    result = await dispatcher.execute(name, arguments)
                    tspan.update(output=result)
                executed[key] = result
            messages.append({"role": "tool", "name": name, "content": result})

        if repeats >= MAX_REPEATED_CALLS:
            log.warning("Model stuck repeating tool calls, forcing final answer")
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Dừng gọi công cụ. Trả lời khách bằng thông tin hiện có; "
                        "nếu thiếu dữ liệu, nói rõ là không có thông tin và mời "
                        "khách liên hệ shop."
                    ),
                }
            )
            with langfuse.start_as_current_observation(
                name="llm.chat",
                as_type="generation",
                input=messages,
                model=settings.llm_model,
            ) as gen:
                final = await llm.chat(messages)
                gen.update(output=final)
            return final

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
