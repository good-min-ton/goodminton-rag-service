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


async def _seed_pants_swamp(pool, make_embedding):
    # 5 pants (all NEARER the query) + 1 far shoe. A naive global filter+LIMIT 4
    # would return 4 pants and drop the shoe; only the per-category quota forces
    # the shoe in and caps pants at the per-category allotment.
    rows = [
        ("101", "Quần cầu lông", {0: 1.0}),
        ("102", "Quần cầu lông", {1: 1.0}),
        ("103", "Quần cầu lông", {2: 1.0}),
        ("104", "Quần cầu lông", {3: 1.0}),
        ("105", "Quần cầu lông", {4: 1.0}),
        ("201", "Giày cầu lông", {50: 1.0}),  # dim the query never touches -> farthest
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


async def test_multi_category_returns_both(truncate_kb, make_embedding):
    pool = truncate_kb
    await _seed_pants_swamp(pool, make_embedding)
    svc = RetrievalService(pool)
    # Query overlaps pants dims 0..4 (all nearer than the shoe on dim 50).
    q = make_embedding({0: 1.0, 1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6})

    # Sanity: >2 nearer pants exist, so the multi-category quota (cap 2) has to
    # deliberately drop some -> proves the assertions below are non-tautological.
    pants_only = await svc.search(q, k=4, categories=["Quần cầu lông"])
    assert len(pants_only) > 2

    chunks = await svc.search(q, k=4, categories=["Quần cầu lông", "Giày cầu lông"])
    cats_present = {c.source_id for c in chunks}
    pants_in_result = cats_present & {"101", "102", "103", "104", "105"}
    # (a) far shoe forced in despite 5 nearer pants -> a naive global top-4 excludes it.
    assert "201" in cats_present
    # (b) pants capped at the per-category quota (top_k=4 / 2 cats = 2).
    assert len(pants_in_result) <= 2


async def test_no_category_preserves_global_search(truncate_kb, make_embedding):
    pool = truncate_kb
    await _seed(pool, make_embedding)
    svc = RetrievalService(pool)
    chunks = await svc.search(make_embedding({0: 1.0}), k=4)
    assert len(chunks) == 4  # unfiltered, current behavior


async def test_category_filter_admits_static_docs(truncate_kb, make_embedding):
    """A category filter carried over from prior conversation state (e.g. the
    user asked about pants, then asks a policy question) must still surface
    static/help docs, which have no category of their own. Only products are
    restricted to the requested categories."""
    pool = truncate_kb
    async with pool.acquire() as conn:
        # Matching-category product -> passes the category filter directly.
        await conn.execute(
            "INSERT INTO kb_chunks (doc_type, source_id, chunk_index, content, metadata, embedding) "
            "VALUES ('product', '101', 0, 'Sản phẩm: P101', '{\"category\": \"Quần cầu lông\"}'::jsonb, $1)",
            make_embedding({0: 1.0}),
        )
        # Different-category product -> must stay excluded.
        await conn.execute(
            "INSERT INTO kb_chunks (doc_type, source_id, chunk_index, content, metadata, embedding) "
            "VALUES ('product', '201', 0, 'Sản phẩm: P201', '{\"category\": \"Giày cầu lông\"}'::jsonb, $1)",
            make_embedding({2: 1.0}),
        )
        # Static/help doc, no category -> must be admitted regardless of the filter.
        await conn.execute(
            "INSERT INTO kb_chunks (doc_type, source_id, chunk_index, content, metadata, embedding) "
            "VALUES ('static', 'policy-1', 0, 'Chính sách bảo hành...', '{}'::jsonb, $1)",
            make_embedding({1: 1.0}),
        )

    svc = RetrievalService(pool)
    chunks = await svc.search(
        make_embedding({0: 1.0}), k=3, categories=["Quần cầu lông"]
    )
    ids = {(c.doc_type, c.source_id) for c in chunks}
    assert ("static", "policy-1") in ids  # not filtered out despite no category
    assert ("product", "101") in ids  # matching category still included
    assert ("product", "201") not in ids  # different-category product excluded
