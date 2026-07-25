# tests/test_similar_service.py
from app.services.similar import _parse_name


def test_parse_name_from_chunk0():
    content = (
        "Sản phẩm: Vợt Yonex Astrox 88D\nThương hiệu: Yonex\nDanh mục: Vợt cầu lông\n"
    )
    assert _parse_name(content) == "Vợt Yonex Astrox 88D"


def test_parse_name_missing_prefix_returns_none():
    assert _parse_name("Thương hiệu: Yonex\nDanh mục: Vợt") is None
    assert _parse_name("") is None
