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
    {
        "type": "function",
        "function": {
            "name": "prepare_order",
            "description": (
                "Tạo ĐƠN HÀNG NHÁP (chưa đặt) đã tính giá + kiểm tra tồn kho để khách xác "
                "nhận trên giao diện. CHỈ gọi khi khách muốn mua/đặt VÀ đã biết variant_id "
                "(PHẢI gọi get_pricing trước để lấy variant_id). Tool này KHÔNG tạo đơn thật."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "maxItems": 20,
                        "description": "Danh sách dòng sản phẩm cần đặt.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "product_id": {
                                    "type": "integer",
                                    "description": "ID sản phẩm (từ danh sách hợp lệ).",
                                },
                                "variant_id": {
                                    "type": "integer",
                                    "description": "ID variant (lấy từ get_pricing).",
                                },
                                "quantity": {
                                    "type": "integer",
                                    "description": "Số lượng, >= 1.",
                                },
                            },
                            "required": ["product_id", "variant_id", "quantity"],
                        },
                    }
                },
                "required": ["items"],
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
            elif name == "prepare_order":
                result = await self._prepare_order(arguments.get("items") or [])
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

    async def _prepare_order(self, items: list[dict]) -> dict:
        """Build a priced, stock-checked order draft. Read-only: reuses
        get_pricing + check_inventory. Rejects (never clamps) invalid
        quantities and drops unknown variants with a warning. Bad int casts
        propagate to execute()'s shared (KeyError, TypeError, ValueError)
        handler."""
        pricing_cache: dict[int, dict] = {}
        lines: list[dict] = []
        warnings: list[str] = []
        total = 0.0

        for raw in items[:20]:  # read-amplification cap
            product_id = int(raw["product_id"])
            variant_id = int(raw["variant_id"])
            quantity = int(raw["quantity"])

            # Reject, never clamp: invalid quantity -> warn + drop (before any read).
            if quantity <= 0 or quantity > 20:
                warnings.append(
                    f"Số lượng {quantity} không hợp lệ cho variant {variant_id}, đã bỏ qua."
                )
                continue

            if product_id not in pricing_cache:
                pricing_cache[product_id] = await self._client.get_pricing(product_id)
            pricing = pricing_cache[product_id]

            variant = next(
                (
                    v
                    for v in pricing.get("variants", [])
                    if v.get("variantId") == variant_id
                ),
                None,
            )
            if variant is None:
                warnings.append(
                    f"variant_id {variant_id} không thuộc sản phẩm {product_id}, đã bỏ qua."
                )
                continue

            sale = variant.get("salePrice")
            unit_price = float(sale if sale is not None else variant.get("price"))
            line_total = unit_price * quantity
            product_name = pricing.get("productName") or ""
            color = variant.get("colorName")
            size = variant.get("sizeName")

            inventory = await self._client.check_inventory(variant_id)
            central_qty = next(
                (
                    row.get("quantity", 0)
                    for row in inventory
                    if row.get("storeName") == settings.central_store_name
                ),
                0,  # no central row -> fail toward out-of-stock, never fail-open
            )
            in_stock = central_qty >= quantity
            if not in_stock:
                label = " ".join(x for x in (color, size) if x)
                warnings.append(
                    f"{product_name} ({label}) chỉ còn {central_qty} tại kho, cần {quantity}."
                )

            total += line_total
            lines.append(
                {
                    "product_id": str(product_id),
                    "variant_id": str(variant_id),
                    "product_name": product_name,
                    "size": size,
                    "color": color,
                    "quantity": quantity,
                    "unit_price": unit_price,
                    "line_total": line_total,
                    "in_stock": in_stock,
                }
            )

        return {
            "items": lines,
            "total": total,
            "currency": "VND",
            "warnings": warnings,
        }
