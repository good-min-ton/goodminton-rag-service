from app.routers.chat import (
    SANITIZE_FALLBACK,
    _recover_tool_call,
    _sanitize_answer,
)

_TOOLS = {
    "get_product_availability",
    "recommend_similar_products",
    "start_order",
}


def test_recover_named_tool_call():
    got = _recover_tool_call(
        '{"name": "get_product_availability", "arguments": {"product_id": 12}}', _TOOLS
    )
    assert got == {"name": "get_product_availability", "arguments": {"product_id": 12}}


def test_recover_nested_function_form():
    got = _recover_tool_call(
        '{"function": {"name": "recommend_similar_products", '
        '"arguments": {"product_id": 45}}}',
        _TOOLS,
    )
    assert got == {
        "name": "recommend_similar_products",
        "arguments": {"product_id": 45},
    }


def test_recover_unknown_tool_name_returns_none():
    assert _recover_tool_call('{"name": "drop_table", "arguments": {}}', _TOOLS) is None


def test_recover_bare_args_no_name_returns_none():
    # the audit's leaked example — no tool name, must NOT be guessed
    assert (
        _recover_tool_call('{"product_id": 164, "size": "M", "quantity": 1}', _TOOLS)
        is None
    )


def test_recover_non_json_returns_none():
    assert _recover_tool_call("Chào bạn, mình có thể giúp gì?", _TOOLS) is None


def test_sanitize_pure_json_returns_fallback():
    assert (
        _sanitize_answer('{"product_id": 164, "size": "M", "quantity": 1}')
        == SANITIZE_FALLBACK
    )


def test_sanitize_fenced_json_only_returns_fallback():
    assert (
        _sanitize_answer('```json\n{"name": "get_product_availability"}\n```')
        == SANITIZE_FALLBACK
    )


def test_sanitize_fence_with_prose_keeps_prose():
    out = _sanitize_answer('```json\n{"x":1}\n```\nDạ vợt còn hàng ạ.')
    assert "Dạ vợt còn hàng ạ." in out
    assert "{" not in out


def test_sanitize_normal_prose_unchanged():
    prose = "Quần Lining 9215 màu đen, size M còn hàng, giá 130.000đ nhé."
    assert _sanitize_answer(prose) == prose


def test_sanitize_empty_returns_fallback():
    assert _sanitize_answer("   ") == SANITIZE_FALLBACK
