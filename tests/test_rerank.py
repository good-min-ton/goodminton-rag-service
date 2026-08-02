from app.core.config import settings
from app.services.rerank import RerankService, _parse_ids


def test_parse_ids_json_array():
    assert _parse_ids('["12", "7"]') == ["12", "7"]


def test_parse_ids_json_with_prose_around():
    assert _parse_ids('Kết quả: ["3", "9"] nhé') == ["3", "9"]


def test_parse_ids_loose_numbers():
    assert _parse_ids("12, 7 và 3") == ["12", "7", "3"]


def test_parse_ids_garbage():
    assert _parse_ids("không có sản phẩm nào phù hợp") == []


class _FakeLLM:
    def __init__(self, out):
        self.out = out
        self.calls = 0

    async def chat(self, messages, **kwargs):
        self.calls += 1
        return self.out


class _BoomLLM:
    async def chat(self, messages, **kwargs):
        raise RuntimeError("boom")


CAND = [
    {"id": "1", "name": "A", "text": "A"},
    {"id": "2", "name": "B", "text": "B"},
    {"id": "3", "name": "C", "text": "C"},
]


async def test_rerank_orders_filters_unknown_and_caps(monkeypatch):
    monkeypatch.setattr(settings, "rerank_enabled", True)
    monkeypatch.setattr(settings, "rerank_mode", "llm")
    svc = RerankService(_FakeLLM('["3","1","99"]'), None)  # 99 not a candidate
    out = await svc.rerank("q", CAND, top_n=2)
    assert out == ["3", "1"]


async def test_rerank_falls_back_to_cosine_on_error(monkeypatch):
    monkeypatch.setattr(settings, "rerank_enabled", True)
    monkeypatch.setattr(settings, "rerank_mode", "llm")
    svc = RerankService(_BoomLLM(), None)
    out = await svc.rerank("q", CAND, top_n=5)
    assert out == ["1", "2", "3"]


async def test_rerank_empty_result_falls_back(monkeypatch):
    monkeypatch.setattr(settings, "rerank_enabled", True)
    monkeypatch.setattr(settings, "rerank_mode", "llm")
    svc = RerankService(_FakeLLM("[]"), None)
    out = await svc.rerank("q", CAND, top_n=5)
    assert out == ["1", "2", "3"]


async def test_rerank_disabled_returns_base_order(monkeypatch):
    monkeypatch.setattr(settings, "rerank_enabled", False)
    svc = RerankService(_FakeLLM('["3"]'), None)
    out = await svc.rerank("q", CAND, top_n=2)
    assert out == ["1", "2"]


class _Resp:
    def __init__(self, scores):
        self._scores = scores

    def raise_for_status(self):
        pass

    def json(self):
        return {"scores": self._scores}


class _FakeHTTP:
    """Minimal async http stub for the bge cross-encoder call."""

    def __init__(self, scores):
        self.scores = scores
        self.url = None

    async def post(self, url, json=None, timeout=None):
        self.url = url
        return _Resp(self.scores)


def test_rerank_mode_defaults_to_bge():
    # The team base defaults to bge (our cross-encoder service) as the canonical
    # ranker; llm listwise stays available as an explicit fallback mode.
    from app.core.config import Settings

    assert Settings.model_fields["rerank_mode"].default == "bge"


async def test_rerank_bge_orders_by_scores_and_falls_back_to_embed_url(monkeypatch):
    monkeypatch.setattr(settings, "rerank_enabled", True)
    monkeypatch.setattr(settings, "rerank_mode", "bge")
    monkeypatch.setattr(settings, "rerank_url", None)  # -> fall back to embed_service_url
    monkeypatch.setattr(settings, "embed_service_url", "http://embed:8003")
    http = _FakeHTTP([0.1, 0.9, 0.5])  # cand 1/2/3 -> ranked by score desc: 2,3,1
    svc = RerankService(None, http)
    out = await svc.rerank("q", CAND, top_n=3)
    assert out == ["2", "3", "1"]
    assert http.url == "http://embed:8003/rerank"  # bge activates on mode alone


async def test_rerank_bge_degrades_to_cosine_when_service_down(monkeypatch):
    class _BoomHTTP:
        async def post(self, *a, **k):
            raise RuntimeError("connect fail")

    monkeypatch.setattr(settings, "rerank_enabled", True)
    monkeypatch.setattr(settings, "rerank_mode", "bge")
    monkeypatch.setattr(settings, "embed_service_url", "http://embed:8003")
    svc = RerankService(None, _BoomHTTP())
    out = await svc.rerank("q", CAND, top_n=5)
    assert out == ["1", "2", "3"]  # degrade to input (cosine) order, never empty
