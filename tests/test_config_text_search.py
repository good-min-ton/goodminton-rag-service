from app.core.config import settings


def test_text_search_max_distance_default_disabled():
    assert settings.text_search_max_distance == 0.0
