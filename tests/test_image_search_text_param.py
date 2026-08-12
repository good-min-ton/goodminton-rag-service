import pytest

from app.services.image_search import ImageSearchService


@pytest.mark.asyncio
async def test_max_distance_param_overrides_settings(
    truncate_image_embeddings, make_embedding_768, monkeypatch
):
    from app.core.config import settings
    from tests.conftest import seed_image_embedding

    pool = truncate_image_embeddings
    await seed_image_embedding(pool, 1, 1, "near", make_embedding_768({0: 1.0}))
    await seed_image_embedding(pool, 2, 2, "far", make_embedding_768({700: 1.0}))
    monkeypatch.setattr(settings, "image_search_max_distance", 0.0)  # settings=off

    svc = ImageSearchService(pool)
    # explicit tight param filters the far product...
    assert await svc.search(
        make_embedding_768({0: 1.0}), top_k=10, max_distance=0.5
    ) == ["1"]
    # ...None falls back to settings (0.0 = off) -> both, image route unchanged.
    assert await svc.search(
        make_embedding_768({0: 1.0}), top_k=10, max_distance=None
    ) == [
        "1",
        "2",
    ]
