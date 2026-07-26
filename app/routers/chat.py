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
MAX_REPEATED_CALLS = 3


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

        answer, tool_products, order_draft = await _run_tool_loop(
            llm_svc, tool_dispatcher, messages
        )
        answer, recommended = _extract_recommended(answer, chunks, tool_products)

        root.update(output=answer)
        return ChatResponse(
            answer=answer,
            sources=_unique_sources(chunks),
            products=recommended,
            order_draft=order_draft,
        )


async def _run_tool_loop(
    llm: LLMService, dispatcher: ToolDispatcher, messages: list[dict]
) -> tuple[str, list[dict], dict | None]:
    """Loop: LLM may call tools; execute, feed back, repeat until text answer.

    Identical calls are cached; after MAX_REPEATED_CALLS repeats the model is
    clearly stuck (e.g. retrying an invalid ID), so we force a final answer
    without tools instead of burning the remaining iterations.

    Returns the answer plus the products surfaced by recommend_similar_products
    ({"id","name"}), so the caller can tie the product cards to the answer.
    """
    executed: dict[tuple[str, str], str] = {}
    tool_products: list[dict] = []
    order_draft: dict | None = None
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
            return msg.get("content") or "", tool_products, order_draft

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
                if name == "recommend_similar_products":
                    _collect_tool_products(result, tool_products)
                elif name == "prepare_order":
                    parsed = _parse_order_draft(result)
                    if parsed is not None:
                        order_draft = parsed  # last successful prepare_order wins
            messages.append({"role": "tool", "name": name, "content": result})

        if repeats >= MAX_REPEATED_CALLS:
            log.warning("Model stuck repeating tool calls, forcing final answer")
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Dừng gọi công cụ. Trả lời khách bằng thông tin hiện có. "
                        "Nếu bạn CHƯA tạo được đơn hàng nháp, nói rõ là chưa tạo "
                        "được đơn và mời khách thử lại — TUYỆT ĐỐI KHÔNG nói đã "
                        "tạo thẻ xác nhận hay đã đặt đơn. Nếu thiếu dữ liệu khác, "
                        "nói rõ là không có thông tin và mời khách liên hệ shop."
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
            return final, tool_products, order_draft

    log.warning("Tool loop hit max iterations (%d)", MAX_TOOL_ITERATIONS)
    return (
        "Xin lỗi, mình không xử lý được yêu cầu này. Vui lòng liên hệ shop.",
        tool_products,
        order_draft,
    )


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


def _chunk_product_name(c: Chunk) -> str:
    """Product name from a chunk whose content starts 'Sản phẩm: <name>'."""
    first_line = c.content.split("\n", 1)[0]
    prefix = "Sản phẩm:"
    return first_line[len(prefix) :].strip() if first_line.startswith(prefix) else ""


_NAME_PREFIXES = (
    "vợt cầu lông ",
    "giày cầu lông ",
    "áo cầu lông ",
    "quần cầu lông ",
    "balo cầu lông ",
    "túi cầu lông ",
    "set vợt cầu lông ",
)


def _name_core(name: str) -> str:
    """Distinctive core of a product name: drop the category prefix and the
    'chính hãng' suffix so a shortened mention in the answer still matches."""
    s = name.strip().lower()
    for p in _NAME_PREFIXES:
        if s.startswith(p):
            s = s[len(p) :]
            break
    for suf in (" chính hãng", " chinh hang"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s.strip()


def _collect_tool_products(result: str, out: list[dict]) -> None:
    """Parse {product_id, name} out of a recommend_similar_products result."""
    try:
        data = json.loads(result)
    except (ValueError, TypeError):
        return
    if not isinstance(data, list):
        return
    seen = {p["id"] for p in out}
    for item in data:
        if (
            isinstance(item, dict)
            and item.get("product_id") is not None
            and item.get("name")
        ):
            pid = str(item["product_id"])
            if pid not in seen:
                seen.add(pid)
                out.append({"id": pid, "name": str(item["name"])})


def _parse_order_draft(result: str) -> dict | None:
    """Parse a prepare_order tool result into an order_draft dict, or None.

    Centralized {"error": ...} payloads and non-JSON strings yield no draft
    (present-or-absent, not inferred from prose like _extract_recommended).
    """
    try:
        data = json.loads(result)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict) or "items" not in data or "error" in data:
        return None
    return data


def _extract_recommended(
    answer: str, chunks: list[Chunk], tool_products: list[dict]
) -> tuple[str, list[str]]:
    """Product ids the answer recommends, matched by name against the products
    it could know about — retrieved chunks + recommend_similar_products results
    — ordered by first mention. Keeps chatbot cards consistent with the text.
    """
    # Build core-name -> id (retrieval first so it wins ties, then tool results)
    core_to_id: dict[str, str] = {}
    for c in chunks:
        if c.doc_type == "product":
            nm = _chunk_product_name(c)
            if nm:
                core_to_id.setdefault(_name_core(nm), c.source_id)
    for p in tool_products:
        nm = p.get("name")
        if nm:
            core_to_id.setdefault(_name_core(nm), p["id"])

    low = answer.lower()
    hits: list[tuple[int, str]] = []
    seen: set[str] = set()
    for core, pid in core_to_id.items():
        pos = low.find(core) if core else -1
        if pos >= 0 and pid not in seen:
            seen.add(pid)
            hits.append((pos, pid))
    hits.sort()
    return answer, [pid for _, pid in hits]


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
