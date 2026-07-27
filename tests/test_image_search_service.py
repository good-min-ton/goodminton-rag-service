import pytest

from app.services.image_search import ImageSearchService


@pytest.mark.asyncio
async def test_search_ranks_by_per_product_min_distance(
    truncate_image_embeddings, make_embedding_768, monkeypatch
):
    from app.core.config import settings
    from tests.conftest import seed_image_embedding

    # Disable the relevance threshold: this test covers MIN-per-product ranking,
    # not the distance cutoff (product 3 here is intentionally far/orthogonal).
    monkeypatch.setattr(settings, "image_search_max_distance", 0.0)
    pool = truncate_image_embeddings
    # product 1: a near image (dim0) AND a far image (dim5) -> MIN keeps it nearest.
    await seed_image_embedding(pool, 1, 1, "a", make_embedding_768({0: 1.0}))
    await seed_image_embedding(pool, 2, 1, "b", make_embedding_768({5: 1.0}))
    # product 2: moderately close.
    await seed_image_embedding(pool, 3, 2, "c", make_embedding_768({0: 0.9, 1: 0.1}))
    # product 3: far.
    await seed_image_embedding(pool, 4, 3, "d", make_embedding_768({5: 1.0}))

    svc = ImageSearchService(pool)
    ids = await svc.search(make_embedding_768({0: 1.0}), top_k=10)

    # Deduped per product, ranked by ascending MIN distance, ids are STRINGS (H1).
    assert ids == ["1", "2", "3"]


@pytest.mark.asyncio
async def test_search_over_fetches_top_k_times_factor(
    truncate_image_embeddings, make_embedding_768, monkeypatch
):
    from app.core.config import settings
    from tests.conftest import seed_image_embedding

    pool = truncate_image_embeddings
    # 5 products, 1 image each, distance increases with pid.
    for pid in range(1, 6):
        await seed_image_embedding(
            pool, pid, pid, f"u{pid}", make_embedding_768({0: 1.0, 700: 0.05 * pid})
        )

    monkeypatch.setattr(settings, "image_search_over_fetch_factor", 3)
    svc = ImageSearchService(pool)
    ids = await svc.search(make_embedding_768({0: 1.0}), top_k=1)

    # top_k=1 but over_fetch = 1*3 = 3 -> returns 3 candidates, not 1 (H8).
    assert ids == ["1", "2", "3"]


@pytest.mark.asyncio
async def test_search_filters_out_far_products_beyond_max_distance(
    truncate_image_embeddings, make_embedding_768, monkeypatch
):
    from app.core.config import settings
    from tests.conftest import seed_image_embedding

    pool = truncate_image_embeddings
    # NEAR product: same dim as query -> distance ~0.
    await seed_image_embedding(pool, 1, 1, "near", make_embedding_768({0: 1.0}))
    # FAR product: orthogonal dim -> distance ~1.0.
    await seed_image_embedding(pool, 2, 2, "far", make_embedding_768({700: 1.0}))

    monkeypatch.setattr(settings, "image_search_max_distance", 0.5)
    svc = ImageSearchService(pool)
    ids = await svc.search(make_embedding_768({0: 1.0}), top_k=10)

    # Far product dropped by the HAVING threshold; only the near one remains.
    assert ids == ["1"]


@pytest.mark.asyncio
async def test_search_returns_empty_for_unrelated_query(
    truncate_image_embeddings, make_embedding_768, monkeypatch
):
    from app.core.config import settings
    from tests.conftest import seed_image_embedding

    pool = truncate_image_embeddings
    await seed_image_embedding(pool, 1, 1, "a", make_embedding_768({0: 1.0}))
    await seed_image_embedding(pool, 2, 2, "b", make_embedding_768({1: 1.0}))

    monkeypatch.setattr(settings, "image_search_max_distance", 0.5)
    svc = ImageSearchService(pool)
    # Orthogonal to every seeded product -> nothing passes the threshold.
    ids = await svc.search(make_embedding_768({700: 1.0}), top_k=10)

    assert ids == []


@pytest.mark.asyncio
async def test_search_disabled_threshold_keeps_old_behavior(
    truncate_image_embeddings, make_embedding_768, monkeypatch
):
    from app.core.config import settings
    from tests.conftest import seed_image_embedding

    pool = truncate_image_embeddings
    await seed_image_embedding(pool, 1, 1, "near", make_embedding_768({0: 1.0}))
    await seed_image_embedding(pool, 2, 2, "far", make_embedding_768({700: 1.0}))

    monkeypatch.setattr(settings, "image_search_max_distance", 0.0)
    svc = ImageSearchService(pool)
    ids = await svc.search(make_embedding_768({0: 1.0}), top_k=10)

    # No filter applied -> far product still returned (old behavior).
    assert ids == ["1", "2"]
