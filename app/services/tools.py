"""Tool definitions and dispatcher for LLM function calling."""

import json
import logging
from typing import Any

from app.services.product_client import ProductClient

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
]


class ToolDispatcher:
    def __init__(self, product_client: ProductClient):
        self._client = product_client

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Run a tool by name and return its result as a JSON string for the LLM."""
        try:
            if name == "get_pricing":
                result = await self._client.get_pricing(int(arguments["product_id"]))
            elif name == "check_inventory":
                result = await self._client.check_inventory(
                    int(arguments["variant_id"])
                )
            else:
                return json.dumps(
                    {"error": f"Unknown tool: {name}"}, ensure_ascii=False
                )

            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            log.exception("Tool %s failed with args %s", name, arguments)
            return json.dumps({"error": str(e)}, ensure_ascii=False)
