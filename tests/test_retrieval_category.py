from app.services.retrieval import RetrievalService


async def _seed(pool, make_embedding):
    # 3 products in category A, 1 in B; distinct embedding dims for deterministic order.
    rows = [
        ("101", "Quần cầu lông", {0: 1.0}),
        ("102", "Quần cầu lông", {1: 1.0}),
        ("103", "Quần cầu lông", {2: 1.0}),
        ("201", "Giày cầu lông", {3: 1.0}),
    ]
    async with pool.acquire() as conn:
        for sid, cat, dims in rows:
            await conn.execute(
                "INSERT INTO kb_chunks (doc_type, source_id, chunk_index, content, metadata, embedding) "
                "VALUES ('product', $1, 0, $2, $3::jsonb, $4)",
                sid,
                f"Sản phẩm: P{sid}",
                f'{{"category": "{cat}"}}',
                make_embedding(dims),
            )


async def test_category_filter_restricts_results(truncate_kb, make_embedding):
    pool = truncate_kb
    await _seed(pool, make_embedding)
    svc = RetrievalService(pool)
    q = make_embedding({3: 1.0})  # closest to the shoe row
    chunks = await svc.search(q, categories=["Quần cầu lông"])
    assert chunks, "expected pants results"
    assert all(
        c.source_id in {"101", "102", "103"} for c in chunks
    )  # no shoe leaked in


async def test_multi_category_returns_both(truncate_kb, make_embedding):
    pool = truncate_kb
    await _seed(pool, make_embedding)
    svc = RetrievalService(pool)
    q = make_embedding({0: 1.0})
    chunks = await svc.search(q, k=4, categories=["Quần cầu lông", "Giày cầu lông"])
    cats_present = {c.source_id for c in chunks}
    assert "201" in cats_present  # the single shoe is present despite 3 closer pants
    assert cats_present & {"101", "102", "103"}  # pants present too


async def test_no_category_preserves_global_search(truncate_kb, make_embedding):
    pool = truncate_kb
    await _seed(pool, make_embedding)
    svc = RetrievalService(pool)
    chunks = await svc.search(make_embedding({0: 1.0}), k=4)
    assert len(chunks) == 4  # unfiltered, current behavior
