# tests/test_llm_overrides.py
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import settings
from app.services.llm import LLMService


def test_description_config_defaults():
    assert settings.description_model is None
    assert settings.description_temperature == 0.5
    assert settings.description_num_predict == 512


@pytest.mark.asyncio
async def test_chat_default_body_unchanged():
    client = MagicMock()
    resp = MagicMock()
    resp.json.return_value = {"message": {"content": "hi"}}
    resp.raise_for_status = MagicMock()
    client.post = AsyncMock(return_value=resp)
    svc = LLMService(client)
    out = await svc.chat([{"role": "user", "content": "x"}])
    assert out == "hi"
    body = client.post.call_args.kwargs["json"]
    assert body["model"] == settings.llm_model
    assert body["options"]["temperature"] == settings.llm_temperature
    assert "num_predict" not in body["options"]


@pytest.mark.asyncio
async def test_chat_overrides_applied():
    client = MagicMock()
    resp = MagicMock()
    resp.json.return_value = {"message": {"content": "copy"}}
    resp.raise_for_status = MagicMock()
    client.post = AsyncMock(return_value=resp)
    svc = LLMService(client)
    await svc.chat(
        [{"role": "user", "content": "x"}],
        temperature=0.5,
        num_predict=256,
        model="qwen2.5:7b",
    )
    body = client.post.call_args.kwargs["json"]
    assert body["model"] == "qwen2.5:7b"
    assert body["options"]["temperature"] == 0.5
    assert body["options"]["num_predict"] == 256
