# app/services/similar.py
from __future__ import annotations

from dataclasses import dataclass

import asyncpg

_NAME_PREFIX = "Sản phẩm: "


class ProductNotIndexedError(Exception):
    """Raised when the source product has no indexed chunks (truly not found).

    Distinct from find_similar returning [] (the product exists but has no
    similar peers), so the router can map only THIS to HTTP 404.
    """


@dataclass
class SimilarResult:
    product_id: str
    name: str | None
    distance: float
    chunk_count: int


def _parse_name(content: str) -> str | None:
    """Best-effort: first line after the 'Sản phẩm: ' prefix in chunk_index=0 text."""
    if not content:
        return None
    first_line = content.split("\n", 1)[0].strip()
    if first_line.startswith(_NAME_PREFIX):
        name = first_line[len(_NAME_PREFIX) :].strip()
        return name or None
    return None


class SimilarProductsService:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool
