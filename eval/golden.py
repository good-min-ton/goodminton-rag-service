"""Golden-set record schema + fail-fast loader (closed category vocabulary)."""

import json
from dataclasses import dataclass

# Exactly the keys of app.services.query_understanding.CATEGORY_KEYWORDS.
CATEGORY_VOCAB: set[str] = {
    "Vợt cầu lông",
    "Giày cầu lông",
    "Quần cầu lông",
    "Áo cầu lông",
    "Dây cước cầu lông",
}
QUERY_TYPES: set[str] = {
    "browse",
    "attribute",
    "spec",
    "typo",
    "multi-category",
    "known-item",
}
SOURCES: set[str] = {"hand", "semi-auto"}


@dataclass
class GoldenRecord:
    id: str
    query: str
    query_type: str
    relevant_source_ids: list[str]
    expected_categories: list[str]
    price_constrained: bool
    source: str
    notes: str = ""


def load_golden(path: str, valid_source_ids: set[str] | None = None):
    kept: list[GoldenRecord] = []
    excluded: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            rid = raw["id"]
            if raw["query_type"] not in QUERY_TYPES:
                raise ValueError(f"{rid}: bad query_type {raw['query_type']!r}")
            if raw["source"] not in SOURCES:
                raise ValueError(f"{rid}: bad source {raw['source']!r}")
            bad_cat = [c for c in raw["expected_categories"] if c not in CATEGORY_VOCAB]
            if bad_cat:
                raise ValueError(f"{rid}: category out of vocab {bad_cat!r}")
            ids = [str(s) for s in raw["relevant_source_ids"]]
            if not ids:
                raise ValueError(f"{rid}: relevant_source_ids empty")
            if valid_source_ids is not None:
                present = [s for s in ids if s in valid_source_ids]
                if not present:
                    excluded.append({"id": rid, "reason": "no relevant in corpus"})
                    continue
                ids = present
            kept.append(
                GoldenRecord(
                    id=rid,
                    query=raw["query"],
                    query_type=raw["query_type"],
                    relevant_source_ids=ids,
                    expected_categories=list(raw["expected_categories"]),
                    price_constrained=bool(raw["price_constrained"]),
                    source=raw["source"],
                    notes=raw.get("notes", ""),
                )
            )
    return kept, excluded
