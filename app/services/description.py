# app/services/description.py
from __future__ import annotations

from app.core.config import settings
from app.core.prompts import (
    DESCRIPTION_SYSTEM_PROMPT,
    DESCRIPTION_USER_TEMPLATE,
    LENGTH_INSTRUCTIONS,
    STYLE_INSTRUCTIONS,
)
from app.services.indexer import strip_html, strip_pricing
from app.services.llm import LLMService
from app.services.product_client import ProductClient

# Approx token budget per length, hard-capped by settings.description_num_predict.
_LENGTH_TOKENS = {"short": 160, "medium": 320, "long": 512}


class DescriptionService:
    def __init__(self, llm: LLMService, product_client: ProductClient) -> None:
        self._llm = llm
        self._products = product_client

    def _format_specs(self, product: dict) -> str:
        specs = product.get("specifications") or []
        parts = [
            f"{s.get('name')}: {s.get('value')}"
            for s in specs
            if s.get("name") and s.get("value")
        ]
        return " | ".join(parts) if parts else "N/A"

    async def generate(
        self, product_id: int, style: str, length: str, keywords: list[str]
    ) -> tuple[str, str]:
        product = await self._products.get_for_rag(product_id)

        source_desc_raw = product.get("description")
        source_desc = (
            strip_pricing(strip_html(source_desc_raw)) if source_desc_raw else "N/A"
        )

        user_content = DESCRIPTION_USER_TEMPLATE.format(
            name=product.get("name") or "N/A",
            brand=product.get("brand") or "N/A",
            category=product.get("category") or "N/A",
            specs=self._format_specs(product),
            source_description=source_desc,
            style_instruction=STYLE_INSTRUCTIONS.get(
                style, STYLE_INSTRUCTIONS["ban_hang"]
            ),
            length_instruction=LENGTH_INSTRUCTIONS.get(
                length, LENGTH_INSTRUCTIONS["medium"]
            ),
            keywords=", ".join(keywords) if keywords else "N/A",
        )
        messages = [
            {"role": "system", "content": DESCRIPTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        effective_model = settings.description_model or settings.llm_model
        num_predict = min(
            _LENGTH_TOKENS.get(length, _LENGTH_TOKENS["medium"]),
            settings.description_num_predict,
        )

        raw = await self._llm.chat(
            messages,
            temperature=settings.description_temperature,
            num_predict=num_predict,
            model=effective_model,
        )
        # Anti-hallucination safety net: strip any leaked VND amount.
        clean = strip_pricing(raw)
        return clean, effective_model
