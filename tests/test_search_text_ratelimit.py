def test_text_bucket_independent_of_image_bucket():
    from app.routers.search import (
        _TEXT_RATE_MAX,
        _hits,
        _text_hits,
        _text_rate_limited,
    )

    _text_hits.clear()
    _hits.clear()
    ip = "1.2.3.4"
    for _ in range(_TEXT_RATE_MAX):
        assert _text_rate_limited(ip) is False
    assert _text_rate_limited(ip) is True  # max+1 blocked
    assert len(_hits) == 0  # image bucket untouched
