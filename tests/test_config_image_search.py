def test_image_search_settings_defaults():
    from app.core.config import settings

    assert settings.embed_service_url == "http://localhost:8001"
    assert settings.image_search_top_k > 0
    assert settings.image_search_over_fetch_factor == 3
    assert settings.image_max_upload_bytes == 8 * 1024 * 1024
