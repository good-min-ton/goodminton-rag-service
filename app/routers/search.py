"""POST /search/image — visual product search (embed -> pgvector -> product_ids)."""

import time
from collections import defaultdict, deque

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.core.config import settings
from app.services.embed_client import EmbedClient, EmbedUnavailable
from app.services.image_search import ImageSearchService

router = APIRouter(prefix="/search", tags=["search"])

# Crude in-memory per-IP rate limit (H5). Unauthenticated + CPU-bound endpoint.
_RATE_WINDOW_SECONDS = 60
_RATE_MAX = 20
_hits: dict[str, deque[float]] = defaultdict(deque)


def _rate_limited(ip: str) -> bool:
    now = time.monotonic()
    dq = _hits[ip]
    while dq and now - dq[0] > _RATE_WINDOW_SECONDS:
        dq.popleft()
    if len(dq) >= _RATE_MAX:
        return True
    dq.append(now)
    return False


@router.post("/image")
async def search_image(http_request: Request, file: UploadFile = File(...)) -> dict:
    ip = http_request.client.host if http_request.client else "unknown"
    if _rate_limited(ip):
        raise HTTPException(status_code=429, detail="Quá nhiều yêu cầu, thử lại sau")

    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="File phải là ảnh")

    data = await file.read()
    if len(data) > settings.image_max_upload_bytes:
        raise HTTPException(status_code=413, detail="Ảnh quá lớn")

    embed: EmbedClient = http_request.app.state.embed
    search_svc: ImageSearchService = http_request.app.state.image_search
    try:
        embedding = await embed.embed_image(data)
    except EmbedUnavailable as exc:  # H7
        raise HTTPException(
            status_code=503, detail="Tìm ảnh đang khởi động, thử lại"
        ) from exc

    product_ids = await search_svc.search(embedding)
    return {"product_ids": product_ids}  # H1
