# tests/test_description_service.py
from unittest.mock import AsyncMock

import pytest

from app.core.config import settings
from app.services.description import DescriptionService


def _product():
    return {
        "name": "Vợt Yonex Astrox 88D",
        "brand": "Yonex",
        "category": "Vợt cầu lông",
        "specifications": [{"name": "Trọng lượng", "value": "83g"}],
        "description": "<p>Vợt nhẹ, giá chỉ 1.200.000đ</p>",
    }


@pytest.mark.asyncio
async def test_generate_builds_grounded_prompt_from_fields():
    llm = AsyncMock()
    llm.chat.return_value = "Mô tả tuyệt vời."
    pc = AsyncMock()
    pc.get_for_rag.return_value = _product()
    svc = DescriptionService(llm=llm, product_client=pc)
    text, model = await svc.generate(1, "ban_hang", "medium", [])
    assert text == "Mô tả tuyệt vời."
    assert model == settings.llm_model
    user_msg = llm.chat.call_args.args[0][1]["content"]
    assert "Vợt Yonex Astrox 88D" in user_msg
    assert "Yonex" in user_msg
    assert "Trọng lượng: 83g" in user_msg
    assert "1.200.000" not in user_msg  # source description price-stripped


@pytest.mark.asyncio
async def test_generate_omits_missing_fields_without_crashing():
    llm = AsyncMock()
    llm.chat.return_value = "ok"
    pc = AsyncMock()
    pc.get_for_rag.return_value = {
        "name": "X",
        "brand": None,
        "category": None,
        "specifications": [],
        "description": None,
    }
    svc = DescriptionService(llm=llm, product_client=pc)
    text, _ = await svc.generate(1, "ban_hang", "medium", [])
    assert text == "ok"
    assert llm.chat.called


@pytest.mark.asyncio
async def test_generate_strips_price_from_llm_output():
    llm = AsyncMock()
    llm.chat.return_value = "Sản phẩm cao cấp. Chỉ 1.500.000đ hôm nay!"
    pc = AsyncMock()
    pc.get_for_rag.return_value = _product()
    svc = DescriptionService(llm=llm, product_client=pc)
    text, _ = await svc.generate(1, "ban_hang", "medium", [])
    assert "1.500.000" not in text


@pytest.mark.asyncio
async def test_generate_applies_style_length_keywords():
    llm = AsyncMock()
    llm.chat.return_value = "ok"
    pc = AsyncMock()
    pc.get_for_rag.return_value = _product()
    svc = DescriptionService(llm=llm, product_client=pc)
    await svc.generate(1, "seo", "long", ["vợt công thủ"])
    user_msg = llm.chat.call_args.args[0][1]["content"]
    assert "vợt công thủ" in user_msg


@pytest.mark.asyncio
async def test_generate_uses_description_temperature_and_num_predict():
    llm = AsyncMock()
    llm.chat.return_value = "ok"
    pc = AsyncMock()
    pc.get_for_rag.return_value = _product()
    svc = DescriptionService(llm=llm, product_client=pc)
    await svc.generate(1, "ban_hang", "short", [])
    kwargs = llm.chat.call_args.kwargs
    assert kwargs["temperature"] == settings.description_temperature
    assert isinstance(kwargs["num_predict"], int)


@pytest.mark.asyncio
async def test_generate_model_fallback_and_override(monkeypatch):
    llm = AsyncMock()
    llm.chat.return_value = "ok"
    pc = AsyncMock()
    pc.get_for_rag.return_value = _product()
    svc = DescriptionService(llm=llm, product_client=pc)
    _, model_default = await svc.generate(1, "ban_hang", "medium", [])
    assert model_default == settings.llm_model
    monkeypatch.setattr(settings, "description_model", "qwen2.5:7b")
    _, model_override = await svc.generate(1, "ban_hang", "medium", [])
    assert model_override == "qwen2.5:7b"
