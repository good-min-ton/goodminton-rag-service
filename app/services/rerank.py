"""Rerank retrieved product candidates so chat cards match the query better.

Two modes:
- "llm": listwise rerank with the chat LLM (Qwen) — no extra infrastructure.
- "bge": call an external bge-reranker cross-encoder service (RERANK_URL).

Both degrade gracefully to the input (cosine) order on any failure, and never
return an empty list when there were candidates.
"""

import json
import logging
import re

from app.core.config import settings

log = logging.getLogger(__name__)

_LLM_PROMPT = (
    "Bạn là bộ xếp hạng độ liên quan. Cho câu hỏi của khách và danh sách sản "
    "phẩm ứng viên (mỗi dòng 'id: tên'), hãy chọn các sản phẩm THỰC SỰ liên "
    "quan tới câu hỏi, xếp theo độ liên quan giảm dần.\n"
    '- CHỈ trả về một mảng JSON các id, ví dụ: ["12", "7"].\n'
    "- Loại bỏ sản phẩm không liên quan; nếu không có sản phẩm nào liên quan, "
    "trả về [].\n"
    "- KHÔNG giải thích, KHÔNG thêm chữ nào ngoài mảng JSON.\n\n"
    "Câu hỏi: {query}\n\nỨng viên:\n{listing}\n\nMảng JSON:"
)

_ID_RE = re.compile(r'"?(\d+)"?')


def _parse_ids(raw: str) -> list[str]:
    """Extract product ids from the model output (JSON array or loose text)."""
    text = raw.strip()
    m = re.search(r"\[.*?\]", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, list):
                return [str(x) for x in data]
        except (ValueError, TypeError):
            pass
    return _ID_RE.findall(text)


class RerankService:
    def __init__(self, llm, http):
        self._llm = llm
        self._http = http

    async def rerank(
        self, query: str, candidates: list[dict], top_n: int | None = None
    ) -> list[str]:
        """candidates: [{"id": str, "name": str, "text": str}]. Returns ranked ids."""
        top_n = top_n or settings.rerank_top_n
        base = [c["id"] for c in candidates]
        if not settings.rerank_enabled or len(candidates) <= 1:
            return base[:top_n]
        try:
            if settings.rerank_mode == "bge":
                ranked = await self._bge(query, candidates)
            else:
                ranked = await self._llm_rerank(query, candidates)
        except Exception:  # noqa: BLE001 — degrade gracefully
            log.exception("rerank failed; falling back to cosine order")
            ranked = base
        valid = {c["id"] for c in candidates}
        out: list[str] = []
        seen: set[str] = set()
        for cid in ranked:
            if cid in valid and cid not in seen:
                seen.add(cid)
                out.append(cid)
        return (out or base)[:top_n]

    async def _llm_rerank(self, query: str, candidates: list[dict]) -> list[str]:
        listing = "\n".join(f"{c['id']}: {c['name']}" for c in candidates)
        prompt = _LLM_PROMPT.format(query=query, listing=listing)
        raw = await self._llm.chat(
            [{"role": "user", "content": prompt}], temperature=0.0, num_predict=120
        )
        return _parse_ids(raw)

    async def _bge(self, query: str, candidates: list[dict]) -> list[str]:
        # Default to the shared embed-service (same one image search uses) when no
        # dedicated rerank_url is configured — the /rerank contract is identical.
        base_url = settings.rerank_url or settings.embed_service_url
        r = await self._http.post(
            f"{base_url}/rerank",
            json={
                "query": query,
                "documents": [c.get("text") or c["name"] for c in candidates],
            },
            timeout=10.0,
        )
        r.raise_for_status()
        scores = r.json()["scores"]
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        return [candidates[i]["id"] for i in order]
