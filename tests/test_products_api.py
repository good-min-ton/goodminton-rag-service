import pytest


@pytest.mark.asyncio
async def test_get_similar_endpoint_200_shape(
    truncate_kb, similar_client, make_embedding
):
    from tests.conftest import seed_chunk

    pool = truncate_kb  # same cached pool similar_client queries
    await seed_chunk(pool, "product", 1, 0, "Sản phẩm: A\n", make_embedding({0: 1.0}))
    await seed_chunk(
        pool, "product", 2, 0, "Sản phẩm: B\n", make_embedding({0: 0.9, 1: 0.1})
    )
    await seed_chunk(pool, "product", 3, 0, "Sản phẩm: C\n", make_embedding({5: 1.0}))

    resp = await similar_client.get("/products/1/similar?limit=3")
    assert resp.status_code == 200
    body = resp.json()
    assert body["product_id"] == "1"
    assert body["count"] == len(body["results"])
    ids = [r["product_id"] for r in body["results"]]
    assert "1" not in ids
    dists = [r["distance"] for r in body["results"]]
    assert dists == sorted(dists)
    for r in body["results"]:
        assert {
            "product_id",
            "name",
            "similarity",
            "distance",
            "chunk_count",
        } <= r.keys()
        # End-to-end arithmetic: the router must wire similarity = 1.0 - distance.
        assert r["similarity"] == pytest.approx(1.0 - r["distance"])


@pytest.mark.asyncio
async def test_get_similar_lone_product_returns_200_empty(
    truncate_kb, similar_client, make_embedding
):
    from tests.conftest import seed_chunk

    # Indexed product with no peers -> 200 + empty results, NOT 404.
    pool = truncate_kb
    await seed_chunk(pool, "product", 1, 0, "Sản phẩm: A\n", make_embedding({0: 1.0}))
    resp = await similar_client.get("/products/1/similar")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 0
    assert body["results"] == []


@pytest.mark.asyncio
async def test_get_similar_not_indexed_returns_404(truncate_kb, similar_client):
    # Empty table -> product has no chunks -> ProductNotIndexedError -> 404.
    resp = await similar_client.get("/products/999999/similar")
    assert resp.status_code == 404
    assert "sản phẩm" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_get_similar_limit_validation_422(stub_similar_client):
    # Query bounds are validated before the service is called -> no DB needed.
    r0 = await stub_similar_client.get("/products/1/similar?limit=0")
    assert r0.status_code == 422
    r_over = await stub_similar_client.get("/products/1/similar?limit=51")
    assert r_over.status_code == 422
