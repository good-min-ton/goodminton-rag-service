"""GET /products/{product_id}/similar — cosine-similarity ranked peer products.

Also: POST /products/{product_id}/description — LLM-generated product copy.
"""

import httpx
from fastapi import APIRouter, HTTPException, Query, Request

from app.core.config import settings
from app.models.schemas import (
    DescriptionRequest,
    DescriptionResponse,
    SimilarProduct,
    SimilarProductsResponse,
)
from app.services.description import DescriptionService
from app.services.similar import ProductNotIndexedError, SimilarProductsService

router = APIRouter(prefix="/products", tags=["products"])


@router.get("/{product_id}/similar")
async def similar(
    product_id: int,
    http_request: Request,
    limit: int = Query(
        default=settings.similar_products_top_k,
        ge=1,
        le=settings.similar_products_max_limit,
    ),
) -> SimilarProductsResponse:
    svc: SimilarProductsService = http_request.app.state.similar
    try:
        results = await svc.find_similar(product_id, limit)
    except ProductNotIndexedError as exc:
        # ONLY the source-missing case is a 404. An empty-but-valid result
        # (indexed product, no peers) falls through to a 200 with results: [].
        raise HTTPException(status_code=404, detail="Không tìm thấy sản phẩm") from exc
    items = [
        SimilarProduct(
            product_id=r.product_id,
            name=r.name,
            similarity=1.0 - r.distance,
            distance=r.distance,
            chunk_count=r.chunk_count,
        )
        for r in results
    ]
    return SimilarProductsResponse(
        product_id=str(product_id), count=len(items), results=items
    )


@router.post("/{product_id}/description")
async def generate_description(
    product_id: int,
    request: DescriptionRequest,
    http_request: Request,
) -> DescriptionResponse:
    desc_svc: DescriptionService = http_request.app.state.description
    try:
        description, model = await desc_svc.generate(
            product_id, request.style, request.length, request.keywords
        )
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            raise HTTPException(
                status_code=404, detail="Không tìm thấy sản phẩm"
            ) from exc
        raise HTTPException(
            status_code=502, detail="Lỗi khi tạo mô tả từ mô hình ngôn ngữ"
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502, detail="Không kết nối được dịch vụ tạo mô tả"
        ) from exc
    return DescriptionResponse(
        product_id=product_id,
        description=description,
        model=model,
        style=request.style,
        length=request.length,
    )
