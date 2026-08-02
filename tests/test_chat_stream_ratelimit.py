from app.routers.chat import _chat_rate_limited, _STREAM_RATE_HITS
from app.core.config import settings


def test_chat_rate_limit_trips_after_max():
    _STREAM_RATE_HITS.clear()
    ip = "1.2.3.4"
    for _ in range(settings.chat_rate_max):
        assert _chat_rate_limited(ip) is False
    assert _chat_rate_limited(ip) is True  # max+1 blocked
    assert _chat_rate_limited("9.9.9.9") is False  # separate ip unaffected
