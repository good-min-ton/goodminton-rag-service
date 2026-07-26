# tests/test_config_similar.py
from app.core.config import settings


def test_similar_products_config_defaults():
    assert settings.similar_products_top_k == 5
    assert settings.similar_products_max_limit == 50
