"""One-off semi-auto golden candidate generator. Commit its OUTPUT (golden.jsonl)
after human review; this script is the reproducible path (K2). LLM temperature=0."""

import unicodedata


def strip_accents(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def leaks_name(query: str, name: str, brand: str) -> bool:
    q = strip_accents(query)
    tokens = [t for t in strip_accents(f"{name} {brand}").split() if len(t) >= 3]
    return any(t in q for t in tokens)


def parse_name_brand(content: str) -> tuple[str, str]:
    """Extract (name, brand) from an indexed product header (see
    app/services/indexer.py build_product_text). Falls back to "" per field
    when its prefix line is absent."""
    name = brand = ""
    for line in content.splitlines():
        if line.startswith("Sản phẩm: "):
            name = line[len("Sản phẩm: ") :].strip()
        elif line.startswith("Thương hiệu: "):
            brand = line[len("Thương hiệu: ") :].strip()
    return name, brand


def stratified_sample(rows: list[dict], per_category: int) -> list[dict]:
    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    picked: list[dict] = []
    for cat, items in by_cat.items():
        picked.extend(items[:per_category])
    return picked


_PERSONA_PROMPT = (
    "Bạn là khách mua cầu lông. Viết 2 câu hỏi tự nhiên bạn có thể hỏi để tìm một "
    "sản phẩm thuộc danh mục '{category}' với đặc điểm: {brief}. TUYỆT ĐỐI KHÔNG "
    "nhắc tên/thương hiệu sản phẩm. Mỗi câu một dòng."
)


async def generate_candidates(
    pool, llm, retriever, per_category: int, prompt_version: str
) -> list[dict]:
    async with pool.acquire() as conn:
        # chunk_index=0 => exactly one row per product; its content is the header.
        rows = await conn.fetch(
            "SELECT source_id, metadata->>'category' AS category, content "
            "FROM kb_chunks WHERE doc_type='product' AND chunk_index=0"
        )
    # Real name/brand/category per product, so the leak filter can actually fire.
    products: dict[str, dict] = {}
    for r in rows:
        if not r["category"]:
            continue
        sid = str(r["source_id"])
        name, brand = parse_name_brand(r["content"])
        products[sid] = {"name": name, "brand": brand, "category": r["category"]}
    sampled = stratified_sample(
        [{"source_id": sid, **info} for sid, info in products.items()],
        per_category,
    )
    candidates: list[dict] = []
    for item in sampled:
        brief = "một mẫu phổ thông, tầm trung"  # paraphrased, NOT raw fields (G3)
        prompt = _PERSONA_PROMPT.format(category=item["category"], brief=brief)
        raw = await llm.chat([{"role": "user", "content": prompt}], temperature=0.0)
        for line in [q.strip("-• ").strip() for q in raw.splitlines() if q.strip()]:
            if not line or leaks_name(line, item["name"], item["brand"]):
                continue
            ranked = await retriever.retrieve(line, 10)
            candidates.append(
                {
                    "query": line,
                    "query_type": "attribute",
                    "relevant_source_ids": [item["source_id"]],
                    "expected_categories": [item["category"]],
                    "price_constrained": False,
                    "source": "semi-auto",
                    # id+name+category so the human reviewer can judge completeness.
                    "review_context": [
                        {
                            "source_id": sid,
                            "name": products.get(sid, {}).get("name", ""),
                            "category": products.get(sid, {}).get("category", ""),
                        }
                        for sid in ranked
                    ],  # user ticks ALL valid (GS1/GS4)
                    "provenance": {"prompt_version": prompt_version},
                }
            )
    return candidates
