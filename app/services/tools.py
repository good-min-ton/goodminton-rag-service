"""Tool definitions and dispatcher for LLM function calling."""

import json
import logging
from typing import Any

import httpx

from app.core.config import settings
from app.services.product_client import ProductClient
from app.services.similar import ProductNotIndexedError, SimilarProductsService

log = logging.getLogger(__name__)


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_pricing",
            "description": (
                "Lấy giá hiện tại và danh sách variants (size, màu, SKU) của một sản phẩm. "
                "Dùng khi user hỏi về giá, sale, các phiên bản size/màu."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "integer",
                        "description": "ID của sản phẩm",
                    }
                },
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_inventory",
            "description": (
                "Kiểm tra tồn kho của một variant (biến thể sản phẩm) tại các chi nhánh. "
                "Dùng khi user hỏi 'còn hàng không', 'có size X không', 'cửa hàng nào còn'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "variant_id": {
                        "type": "integer",
                        "description": "ID của variant (lấy từ get_pricing trước)",
                    }
                },
                "required": ["variant_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_similar_products",
            "description": (
                "Gợi ý các sản phẩm tương tự với một sản phẩm cho trước dựa trên "
                "nội dung/đặc điểm (tên, thương hiệu, danh mục, thông số, mô tả)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "integer",
                        "description": "ID của sản phẩm cần tìm sản phẩm tương tự.",
                    }
                },
                "required": ["product_id"],
            },
        },
    },
]


class ToolDispatcher:
    def __init__(
        self, product_client: ProductClient, similar: SimilarProductsService
    ) -> None:
        self._client = product_client
        self._similar = similar

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Run a tool by name and return its result as a JSON string for the LLM."""
        try:
            if name == "get_pricing":
                result = await self._client.get_pricing(int(arguments["product_id"]))
            elif name == "check_inventory":
                result = await self._client.check_inventory(
                    int(arguments["variant_id"])
                )
            elif name == "recommend_similar_products":
                try:
                    results = await self._similar.find_similar(
                        int(arguments["product_id"]), settings.similar_products_top_k
                    )
                except ProductNotIndexedError:
                    return json.dumps(
                        {
                            "error": "Không tìm thấy sản phẩm",
                            "product_id": arguments.get("product_id"),
                        },
                        ensure_ascii=False,
                    )
                payload = [
                    {
                        "product_id": r.product_id,
                        "name": r.name,
                        "similarity": 1.0 - r.distance,
                        "distance": r.distance,
                        "chunk_count": r.chunk_count,
                    }
                    for r in results
                ]
                return json.dumps(payload, ensure_ascii=False)
            else:
                return json.dumps(
                    {"error": f"Unknown tool: {name}"}, ensure_ascii=False
                )

            return json.dumps(result, ensure_ascii=False)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                log.warning("Tool %s: not found, args %s", name, arguments)
                return json.dumps(
                    {
                        "error": (
                            f"Không tồn tại dữ liệu cho {arguments}. "
                            "Chỉ dùng ID có trong danh sách được cung cấp trong ngữ cảnh. "
                            "Nếu không có ID phù hợp, hãy trả lời rằng bạn không tìm thấy sản phẩm."
                        )
                    },
                    ensure_ascii=False,
                )
            log.exception("Tool %s failed with args %s", name, arguments)
            return json.dumps(
                {"error": "Hệ thống tra cứu tạm thời gặp lỗi."}, ensure_ascii=False
            )
        except (KeyError, TypeError, ValueError):
            log.warning("Tool %s: invalid args %s", name, arguments)
            return json.dumps(
                {"error": f"Tham số không hợp lệ: {arguments}"}, ensure_ascii=False
            )
        except Exception:
            log.exception("Tool %s failed with args %s", name, arguments)
            return json.dumps(
                {"error": "Hệ thống tra cứu tạm thời gặp lỗi."}, ensure_ascii=False
            )
