from app.models.schemas import SimilarProduct, SimilarProductsResponse


def test_similarity_is_one_minus_distance():
    item = SimilarProduct(
        product_id="2", name="B", similarity=1.0 - 0.1, distance=0.1, chunk_count=3
    )
    assert item.similarity == 0.9
    resp = SimilarProductsResponse(product_id="1", count=1, results=[item])
    assert resp.results[0].distance == 0.1
    assert resp.count == 1
