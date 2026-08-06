"""Collapse chunk-level retrieval rows to a globally-ranked list of distinct
product source_ids (metrics operate on products, not chunks)."""


def rank_products(chunks, k: int) -> list[str]:
    ordered = sorted(
        chunks, key=lambda c: (c.distance, str(c.source_id), c.chunk_index)
    )
    out: list[str] = []
    seen: set[str] = set()
    for c in ordered:
        sid = str(c.source_id)
        if sid in seen:
            continue
        seen.add(sid)
        out.append(sid)
        if len(out) >= k:
            break
    return out
