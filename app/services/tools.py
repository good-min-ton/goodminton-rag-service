"""Tool definitions and dispatcher for LLM function calling."""

import asyncio
import json
import logging
from typing import Any

import httpx

from app.core.config import settings
from app.services.product_client import ProductClient
from app.services.similar import ProductNotIndexedError, SimilarProductsService

log = logging.getLogger(__name__)

# Read-amplification cap: one availability lookup fans out to one inventory call
# per variant. Real products have a handful; this bounds the pathological case.
MAX_VARIANTS_PER_LOOKUP = 20


def _split_stock(rows: list[dict]) -> dict:
    """Split one variant's inventory rows into orderable vs walk-in stock.

    The central store is identified by the `isCentral` flag shop-api sends on
    each row -- the same source of truth its own checkout uses. This used to be a
    store-NAME comparison against a config value, which an ordinary rename or a
    central-store promotion silently broke: every variant then read as zero
    orderable stock, which surfaced as an order card whose button never enabled.

    No central row at all means zero orderable, never "unknown": failing toward
    out-of-stock keeps the bot from promising what checkout cannot deliver.
    """
    orderable = 0
    branches: list[dict] = []
    for row in rows:
        quantity = row.get("quantity") or 0
        if row.get("isCentral"):
            orderable = quantity
        elif quantity > 0:
            branches.append(
                {
                    "storeId": row.get("storeId"),
                    "storeName": row.get("storeName"),
                    "quantity": quantity,
                }
            )
    return {"orderable": orderable, "branches": branches}


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_product_availability",
            "description": (
                "Lấy giá VÀ tồn kho của một sản phẩm trong MỘT lần gọi. Trả về mọi "
                "variant (size, màu, SKU, giá, giá sale) kèm số lượng còn tại từng "
                "chi nhánh. Dùng khi user hỏi giá, sale, các phiên bản size/màu, "
                "'còn hàng không', 'có size X không', 'cửa hàng nào còn'. "
                "KHÔNG cần gọi thêm tool nào khác để biết tồn kho."
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
            "name": "start_order",
            "description": (
                "Mở BẢNG CHỌN để khách tự bấm chọn size, màu và số lượng ngay trên "
                "giao diện. GỌI NGAY khi khách muốn mua/đặt một sản phẩm — KHÔNG "
                "cần hỏi size hay màu trước, bảng chọn đã hiển thị sẵn mọi lựa "
                "chọn còn hàng. Sau khi gọi, chỉ cần mời khách chọn trên bảng."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "integer",
                        "description": "ID sản phẩm khách muốn mua.",
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
            if name == "get_product_availability":
                result = await self._availability(int(arguments["product_id"]))
            elif name == "start_order":
                result = await self._start_order(int(arguments["product_id"]))
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

    async def _availability(self, product_id: int) -> dict:
        """Pricing plus stock for every variant, in one tool call.

        Replaces the old get_pricing -> check_inventory pair. The model had to
        call the first to discover variant ids before it could call the second,
        which cost a whole extra LLM round trip on the most common question
        ("how much is it, is it in stock?"). Fanning the inventory reads out here
        trades that turn for N internal HTTP calls on the Docker network, which
        are orders of magnitude cheaper and run concurrently.

        Stock is split in two, because the two halves mean different things:

        - `orderable` is the central store's quantity. An ONLINE order is always
          fulfilled from there (shop-api's createOnlineOrder deducts from
          findCentralStore), so this is the only number that decides whether an
          order can be placed. Reporting a cross-store total here would have the
          bot promise stock the checkout cannot draw on.
        - `branches` lists the other stores holding stock. Walk-in only, so it is
          advice ("it is in stock at Q7"), never a basis for ordering.
        """
        pricing = await self._client.get_pricing(product_id)
        variants = pricing.get("variants") or []
        capped = variants[:MAX_VARIANTS_PER_LOOKUP]

        # Any failure propagates to execute()'s handlers. These calls hit the same
        # service that just served the pricing read, so a partial failure is far
        # less likely than a total one, and a half-known stock picture would be
        # worse than a clean "lookup failed".
        stocks = await asyncio.gather(
            *(self._client.check_inventory(int(v["variantId"])) for v in capped)
        )

        out: list[dict] = []
        for variant, rows in zip(capped, stocks, strict=True):
            out.append({**variant, **_split_stock(rows)})

        result = {
            "productId": pricing.get("productId"),
            "productName": pricing.get("productName"),
            "variants": out,
        }
        if len(variants) > len(capped):
            result["note"] = (
                f"Chỉ hiển thị {len(capped)}/{len(variants)} variant đầu tiên."
            )
        return result

    async def _start_order(self, product_id: int) -> dict:
        """Build the picker payload: every variant of one product, priced and
        stock-checked, for the customer to choose from on the interface.

        This exists to take variant selection away from the LLM. Asking "which
        size?" in prose cost a generation per question and then required the
        model to map a free-text reply back to a variant_id -- the step that
        produced wrong-variant orders. Here the model only names the product.

        Out-of-stock variants are kept, with orderable = 0. Dropping them would
        tell the customer the size does not exist; the frontend greys them out
        and can point at a branch that has one.
        """
        pricing = await self._client.get_pricing(product_id)
        variants = (pricing.get("variants") or [])[:MAX_VARIANTS_PER_LOOKUP]

        stocks = await asyncio.gather(
            *(self._client.check_inventory(int(v["variantId"])) for v in variants)
        )

        options: list[dict] = []
        for variant, rows in zip(variants, stocks, strict=True):
            stock = _split_stock(rows)
            sale = variant.get("salePrice")
            unit_price = float(sale if sale is not None else variant.get("price") or 0)
            options.append(
                {
                    "variant_id": str(variant.get("variantId")),
                    "size": variant.get("sizeName"),
                    "color": variant.get("colorName"),
                    "unit_price": unit_price,
                    "orderable": stock["orderable"],
                    "branches": [
                        {
                            "store_id": b.get("storeId"),
                            "store_name": b.get("storeName"),
                            "quantity": b["quantity"],
                        }
                        for b in stock["branches"]
                    ],
                }
            )

        return {
            "product_id": str(pricing.get("productId") or product_id),
            "product_name": pricing.get("productName") or "",
            "currency": "VND",
            "options": options,
        }
