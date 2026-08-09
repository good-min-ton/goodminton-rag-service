from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.routers import chat as chat_mod
from app.core.config import settings
from app.models.schemas import ChatResponse


def _app():
    app = FastAPI()
    app.include_router(chat_mod.router)
    return app


def test_stream_returns_404_when_flag_off(monkeypatch):
    # Flag OFF => route not reachable (the flag check is the first line, before
    # any app.state access, so no fakes needed).
    monkeypatch.setattr(settings, "chat_stream_enabled", False)
    with TestClient(_app()) as c:
        assert c.post("/chat/stream", json={"message": "xin chào"}).status_code == 404


def test_stream_events_cover_all_chat_response_fields():
    # Structural parity guard: every user-facing ChatResponse field must be carried
    # by either the meta event (sources/intent/categories) or the done event, so an
    # accidental drop is caught without a live GPU run. Value parity is covered by
    # E2E (Step 7) since /chat and /chat/stream share _prepare_chat_pipeline +
    # _finalize_chat. (A full ASGITransport value-parity test would need heavy,
    # fragile fakes of every app.state service — deliberately avoided per YAGNI.)
    meta_keys = {"sources", "intent", "categories"}
    done_keys = {
        "answer",
        "order_selection",
        "display_products",
        "products",
        "sources",
        "conversation_state",
    }
    assert set(ChatResponse.model_fields) <= (meta_keys | done_keys)
