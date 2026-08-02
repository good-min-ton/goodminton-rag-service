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
