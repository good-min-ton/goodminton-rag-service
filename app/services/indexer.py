"""Product indexer — fetches product data from shop-api, embeds, and upserts kb_chunks."""

import logging
import re

import asyncpg
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.services.embedding import EmbeddingService
from app.services.product_client import ProductClient

log = logging.getLogger(__name__)

# Strip numeric VND amounts only — "1,200,000đ", "1.200.000 VND", "1200000đ"
# Avoid greedy phrase matching that would gut the description.
_PRICE_PATTERNS = [
    re.compile(r"\d{1,3}(?:[.,]\d{3})+\s*(?:đ|vnđ|vnd)?", re.IGNORECASE),
    re.compile(r"\b\d{4,}\s*(?:đ|vnđ|vnd)\b", re.IGNORECASE),
]


def strip_html(html: str | None) -> str:
    if not html:
        return ""
    return BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True)


def strip_pricing(text: str) -> str:
    """Remove VND amounts / 'giá ...' phrases — those belong to live tool calls."""
    for pat in _PRICE_PATTERNS:
        text = pat.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def build_product_text(product: dict) -> str:
    specs = product.get("specifications") or []
    specs_text = (
        " | ".join(f"{s['name']}: {s['value']}" for s in specs) if specs else "N/A"
    )
    description = strip_pricing(strip_html(product.get("description")))
    return (
        f"Sản phẩm: {product['name']}\n"
        f"Thương hiệu: {product['brand']}\n"
        f"Danh mục: {product['category']}\n"
        f"Thông số: {specs_text}\n"
        f"Mô tả: {description}"
    )


class ProductIndexer:
    def __init__(
        self,
        pool: asyncpg.Pool,
        embedding: EmbeddingService,
        product_client: ProductClient,
    ):
        self._pool = pool
        self._embedding = embedding
        self._client = product_client
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.product_chunk_size,
            chunk_overlap=settings.product_chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
        )

    async def index_product(self, product_id: int) -> int:
        """Fetch + chunk + embed + atomic replace. Returns chunk count."""
        product = await self._client.get_for_rag(product_id)
        text = build_product_text(product)
        chunks = self._splitter.split_text(text)
        source_id = str(product_id)

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "DELETE FROM kb_chunks WHERE doc_type='product' AND source_id=$1",
                    source_id,
                )
                for idx, chunk in enumerate(chunks):
                    embedding = await self._embedding.embed(chunk)
                    await conn.execute(
                        """
                        INSERT INTO kb_chunks
                            (doc_type, source_id, chunk_index, content, embedding)
                        VALUES ('product', $1, $2, $3, $4)
                        """,
                        source_id,
                        idx,
                        chunk,
                        embedding,
                    )

        log.info("Indexed product %s: %d chunks", product_id, len(chunks))
        return len(chunks)

    async def delete_product(self, product_id: int) -> int:
        source_id = str(product_id)
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM kb_chunks WHERE doc_type='product' AND source_id=$1",
                source_id,
            )
        log.info("Deleted product %s chunks (%s)", product_id, result)
        # asyncpg returns "DELETE N" — return N
        try:
            return int(result.split()[-1])
        except (ValueError, IndexError):
            return 0
