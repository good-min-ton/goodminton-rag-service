"""POST /chat — RAG chat endpoint with retrieval + LLM + tool calling."""

import json
import logging
import re

from fastapi import APIRouter, HTTPException, Request
from langfuse import propagate_attributes

from app.core.config import settings
from app.core.prompts import SYSTEM_PROMPT
from app.core.tracing import langfuse
from app.models.schemas import ChatRequest, ChatResponse, SourceRef
from app.services.embedding import EmbeddingService
from app.services.llm import LLMService
from app.services.order_flow import (
    BROWSING,
    next_order_status,
    order_directive,
    suppresses_recommendations,
)
from app.services.retrieval import Chunk, RetrievalService
from app.services.tools import TOOL_SCHEMAS, ToolDispatcher

log = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

MAX_TOOL_ITERATIONS = 10
MAX_REPEATED_CALLS = 3

SANITIZE_FALLBACK = "Xin lỗi, bạn cho mình xin lại yêu cầu một chút nhé?"
_TOOL_NAMES: set[str] = {s["function"]["name"] for s in TOOL_SCHEMAS}


def _recover_tool_call(content: str, tool_names: set[str]) -> dict | None:
    """Parse a tool call the model emitted as JSON text in `content` (3B often
    does this instead of the structured tool_calls field). Returns
    {"name", "arguments"} ONLY when the JSON names a known tool with an args
    object; never guesses a tool from bare args. Returns None otherwise."""
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:]
        text = text.strip()
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    fn = data["function"] if isinstance(data.get("function"), dict) else data
    name = fn.get("name")
    args = fn.get("arguments")
    if name in tool_names and isinstance(args, dict):
        return {"name": name, "arguments": args}
    return None


def _sanitize_answer(text: str) -> str:
    """Strip a predominantly-JSON answer so tool-call/JSON text never reaches the
    user. Conservative: only touches answers that start with '{' or a code fence
    (normal prose, incl. prices, is returned unchanged). Empty result -> fallback."""
    stripped = text.strip()
    if not stripped:
        return SANITIZE_FALLBACK
    if not (stripped.startswith("{") or stripped.startswith("```")):
        return text
    try:
        json.loads(stripped)
        return SANITIZE_FALLBACK  # the whole answer is a JSON blob
    except (ValueError, TypeError):
        pass
    cleaned = re.sub(r"```(?:json)?.*?```", "", stripped, flags=re.DOTALL).strip()
    return cleaned if cleaned else SANITIZE_FALLBACK


@router.post("")
async def chat(request: ChatRequest, http_request: Request) -> ChatResponse:
    embedding_svc: EmbeddingService = http_request.app.state.embedding
    retrieval_svc: RetrievalService = http_request.app.state.retrieval
    llm_svc: LLMService = http_request.app.state.llm
    tool_dispatcher: ToolDispatcher = http_request.app.state.tool_dispatcher
    state_store = http_request.app.state.conversation_state
    qu_svc = http_request.app.state.query_understanding

    query = request.message.strip()
    if len(query) < settings.min_query_length:
        raise HTTPException(status_code=400, detail="Câu hỏi quá ngắn")

    with (
        propagate_attributes(session_id=request.session_id or None, trace_name="chat"),
        langfuse.start_as_current_observation(
            name="chat", as_type="span", input=query
        ) as root,
    ):
        state = await state_store.load(request.session_id)
        qu = await qu_svc.analyze(query, state)
        just_placed = (
            request.order_placed_id is not None
            and request.order_placed_id != state.last_confirmed_order_id
        )
        incoming_status = next_order_status(
            state.order_status,
            qu,
            order_draft_emitted=False,
            order_just_placed=just_placed,
        )

        with langfuse.start_as_current_observation(
            name="embed",
            as_type="embedding",
            input=qu.retrieval_query,
            model=settings.embedding_model,
        ):
            query_vec = await embedding_svc.embed(qu.retrieval_query)

        with langfuse.start_as_current_observation(
            name="retrieval", as_type="retriever", input=qu.retrieval_query
        ) as rspan:
            chunks = await retrieval_svc.search(
                query_vec, categories=qu.categories or None
            )
            rspan.update(
                output=[
                    {"doc_type": c.doc_type, "source_id": c.source_id} for c in chunks
                ]
            )

        context = _format_context(chunks)
        catalog = _extract_product_catalog(chunks)

        system_content = SYSTEM_PROMPT.format(context=context)
        if catalog:
            listing = "\n".join(
                f"- {pid}: {name}" if name else f"- {pid}" for pid, name in catalog
            )
            system_content += (
                "\n\nDanh sách sản phẩm hợp lệ để gọi tool. Chọn ĐÚNG product_id ứng "
                "với TÊN sản phẩm khách hỏi; KHÔNG nhắc ID trong câu trả lời, KHÔNG "
                "dùng ID ngoài danh sách này:\n" + listing
            )
        else:
            system_content += (
                "\n\nNgữ cảnh không chứa sản phẩm nào. KHÔNG gọi tool với ID tự nghĩ "
                "ra; nếu khách hỏi giá hoặc tồn kho, trả lời rằng bạn không tìm thấy "
                "sản phẩm phù hợp."
            )

        system_content += order_directive(incoming_status)

        messages: list[dict] = [{"role": "system", "content": system_content}]
        for m in request.chat_history:
            messages.append({"role": m.role, "content": m.content})
        messages.append({"role": "user", "content": query})

        answer, tool_products, order_draft = await _run_tool_loop(
            llm_svc, tool_dispatcher, messages
        )

        order_draft_emitted = order_draft is not None
        final_status = next_order_status(
            state.order_status, qu, order_draft_emitted, order_just_placed=just_placed
        )

        display = _structured_display_products(
            chunks, tool_products, settings.chat_display_products_max
        )
        if suppresses_recommendations(final_status) or not qu.product_query:
            display = []  # no new-product cards while an order is pending/placed
        elif qu.price_preference == "cheapest" and display:
            display = await _sort_by_price(http_request.app.state.http, display)

        # Remember the product under order; clear it when the user moves on to browsing.
        if order_draft_emitted:
            items = order_draft.get("items") or []
            if items:
                try:
                    state.selected_product_id = int(items[0]["product_id"])
                except (KeyError, ValueError, TypeError):
                    pass
        if final_status == BROWSING:
            state.selected_product_id = None

        # Persist scope + order status for the next turn.
        state.categories = qu.categories or state.categories
        state.intent = qu.intent
        state.price_preference = qu.price_preference
        if just_placed:
            state.last_confirmed_order_id = request.order_placed_id
        state.order_status = final_status
        await state_store.save(request.session_id, state)

        root.update(output=answer)
        return ChatResponse(
            answer=answer,
            sources=_unique_sources(chunks),
            products=[str(i) for i in display],  # legacy mirror
            order_draft=order_draft,
            intent=qu.intent,
            categories=qu.categories,
            display_products=display,
            conversation_state=state,
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
            content = msg.get("content") or ""
            recovered = _recover_tool_call(content, _TOOL_NAMES)
            if recovered is None:
                return _sanitize_answer(content), tool_products, order_draft
            # 3B emitted the call as text — honor it: synthesize a structured call
            # and fall through to the normal execution block below.
            tool_calls = [{"function": recovered}]
            msg = {"role": "assistant", "content": "", "tool_calls": tool_calls}

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
            return _sanitize_answer(final), tool_products, order_draft

    log.warning("Tool loop hit max iterations (%d)", MAX_TOOL_ITERATIONS)
    return (
        "Xin lỗi, mình không xử lý được yêu cầu này. Vui lòng liên hệ shop.",
        tool_products,
        order_draft,
    )


def _structured_display_products(
    chunks: list[Chunk], tool_products: list[dict], cap: int
) -> list[int]:
    """The product ids to render as cards, from STRUCTURED sources (not prose):
    recommend_similar_products results first, then retrieved product chunks, in
    order, deduped, numeric-only, capped. Replaces the substring scrape."""
    ordered: list[str] = [p["id"] for p in tool_products if p.get("id")]
    ordered += [c.source_id for c in chunks if c.doc_type == "product"]
    out: list[int] = []
    seen: set[int] = set()
    for sid in ordered:
        try:
            n = int(sid)
        except (TypeError, ValueError):
            continue
        if n > 0 and n not in seen:
            seen.add(n)
            out.append(n)
        if len(out) >= cap:
            break
    return out


async def _sort_by_price(http_client, ids: list[int]) -> list[int]:
    """Sort candidate ids ascending by live price (shop-api). On any failure keep
    the original order — never empty the list because pricing was unavailable."""
    from app.services.product_client import ProductClient

    client = ProductClient(http_client)
    priced: list[tuple[float, int]] = []
    for pid in ids:
        try:
            data = await client.get_pricing(pid)
            variants = data.get("variants") or []
            prices = [
                v.get("salePrice") if v.get("salePrice") is not None else v.get("price")
                for v in variants
            ]
            prices = [p for p in prices if p is not None]
            priced.append((min(prices) if prices else float("inf"), pid))
        except Exception:  # noqa: BLE001 — degrade gracefully
            priced.append((float("inf"), pid))
    priced.sort(key=lambda t: t[0])
    return [pid for _, pid in priced]


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
    (present-or-absent, not inferred from prose).
    """
    try:
        data = json.loads(result)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict) or "items" not in data or "error" in data:
        return None
    return data


def _extract_product_catalog(chunks: list[Chunk]) -> list[tuple[str, str]]:
    """Unique (product_id, name) pairs in retrieval order — tool-calling hints.

    Pairing each id with its name lets the model pick the id that matches the
    product the user named, instead of guessing from a bare id list.
    """
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for c in chunks:
        if c.doc_type != "product":
            continue
        if c.source_id in seen:
            continue
        seen.add(c.source_id)
        out.append((c.source_id, _chunk_product_name(c)))
    return out
