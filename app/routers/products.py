"""GET /products/{product_id}/similar — cosine-similarity ranked peer products."""

from fastapi import APIRouter, HTTPException, Query, Request

from app.core.config import settings
from app.models.schemas import SimilarProduct, SimilarProductsResponse
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
