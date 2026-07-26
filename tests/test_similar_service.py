# tests/test_similar_service.py
import pytest

from app.services.similar import (
    ProductNotIndexedError,
    SimilarProductsService,
    _parse_name,
)


def test_parse_name_from_chunk0():
    content = (
        "Sản phẩm: Vợt Yonex Astrox 88D\nThương hiệu: Yonex\nDanh mục: Vợt cầu lông\n"
    )
    assert _parse_name(content) == "Vợt Yonex Astrox 88D"


def test_parse_name_missing_prefix_returns_none():
    assert _parse_name("Thương hiệu: Yonex\nDanh mục: Vợt") is None
    assert _parse_name("") is None


@pytest.mark.asyncio
async def test_find_similar_excludes_source_and_ranks(truncate_kb, make_embedding):
    from tests.conftest import seed_chunk

    pool = truncate_kb
    # A source points mostly along dim 0.
    await seed_chunk(pool, "product", 1, 0, "Sản phẩm: A\n", make_embedding({0: 1.0}))
    # B very close to A (dim 0), C near-orthogonal (dim 5).
    await seed_chunk(
        pool, "product", 2, 0, "Sản phẩm: B\n", make_embedding({0: 0.9, 1: 0.1})
    )
    await seed_chunk(pool, "product", 3, 0, "Sản phẩm: C\n", make_embedding({5: 1.0}))
    # A static doc close to A must be excluded.
    await seed_chunk(
        pool, "static", "docs/x.md", 0, "irrelevant", make_embedding({0: 1.0})
    )

    svc = SimilarProductsService(pool)
    results = await svc.find_similar(1, limit=10)

    ids = [r.product_id for r in results]
    assert "1" not in ids  # source excluded
    assert "docs/x.md" not in ids  # static excluded
    assert ids == ["2", "3"]  # B closer than C
    assert results[0].name == "B"
    assert results[0].distance <= results[1].distance


@pytest.mark.asyncio
async def test_find_similar_dedups_multichunk_and_counts(truncate_kb, make_embedding):
    from tests.conftest import seed_chunk

    pool = truncate_kb
    await seed_chunk(pool, "product", 1, 0, "Sản phẩm: A\n", make_embedding({0: 1.0}))
    await seed_chunk(pool, "product", 2, 0, "Sản phẩm: B\n", make_embedding({0: 1.0}))
    await seed_chunk(pool, "product", 2, 1, "extra chunk", make_embedding({0: 1.0}))
    await seed_chunk(pool, "product", 2, 2, "extra chunk 2", make_embedding({0: 1.0}))

    svc = SimilarProductsService(pool)
    results = await svc.find_similar(1, limit=10)
    assert len(results) == 1
    assert results[0].product_id == "2"
    assert results[0].chunk_count == 3


@pytest.mark.asyncio
async def test_find_similar_respects_limit(truncate_kb, make_embedding):
    from tests.conftest import seed_chunk

    pool = truncate_kb
    await seed_chunk(pool, "product", 1, 0, "Sản phẩm: A\n", make_embedding({0: 1.0}))
    for pid in range(2, 7):
        await seed_chunk(
            pool,
            "product",
            pid,
            0,
            f"Sản phẩm: P{pid}\n",
            make_embedding({0: 1.0 / pid}),
        )
    svc = SimilarProductsService(pool)
    results = await svc.find_similar(1, limit=2)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_find_similar_unknown_product_raises(truncate_kb):
    svc = SimilarProductsService(truncate_kb)
    with pytest.raises(ProductNotIndexedError):
        await svc.find_similar(999, limit=5)


@pytest.mark.asyncio
async def test_find_similar_lone_product_returns_empty(truncate_kb, make_embedding):
    from tests.conftest import seed_chunk

    # Source exists (indexed) but there are NO other products -> [], NOT an error.
    pool = truncate_kb
    await seed_chunk(pool, "product", 1, 0, "Sản phẩm: A\n", make_embedding({0: 1.0}))
    svc = SimilarProductsService(pool)
    assert await svc.find_similar(1, limit=5) == []


@pytest.mark.asyncio
async def test_find_similar_centroid_of_multichunk(truncate_kb, make_embedding):
    from tests.conftest import seed_chunk

    pool = truncate_kb
    # A has 2 chunks in different directions -> centroid points between dims 0 and 1.
    await seed_chunk(pool, "product", 1, 0, "Sản phẩm: A\n", make_embedding({0: 1.0}))
    await seed_chunk(pool, "product", 1, 1, "second", make_embedding({1: 1.0}))
    # B aligned with the mean (both dims); C aligned with only one of A's chunks.
    await seed_chunk(
        pool, "product", 2, 0, "Sản phẩm: B\n", make_embedding({0: 1.0, 1: 1.0})
    )
    await seed_chunk(pool, "product", 3, 0, "Sản phẩm: C\n", make_embedding({0: 1.0}))
    svc = SimilarProductsService(pool)
    results = await svc.find_similar(1, limit=2)
    assert results[0].product_id == "2"  # mean-aligned ranks above single-dim
