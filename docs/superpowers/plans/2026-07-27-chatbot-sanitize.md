# Chatbot Tool-Result Sanitize Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop raw JSON / tool-call syntax from leaking into the chatbot's answer, and recover the tool call the 3B model emits as content text — closing audit problem P4.

**Architecture:** Two pure helpers in `app/routers/chat.py` — `_recover_tool_call` (parse a content-embedded tool call the model emitted as text, only when it names a known tool) and `_sanitize_answer` (strip a predominantly-JSON answer, falling back to a polite message) — wired into `_run_tool_loop`'s no-tool-calls branch and stuck-fallback, plus an anti-echo rule in the system prompt.

**Tech Stack:** FastAPI, Python (`uv`). RAG-only; builds on Specs 1+2.

**Spec:** `docs/superpowers/specs/2026-07-27-chatbot-sanitize-design.md`

## Global Constraints

- Branch: RAG `feat/chatbot-sanitize` (already created off `feat/chatbot-order-flow`, checked out). Stacked; rebase order Foundation → Order-flow → Sanitize.
- RAG-only. Frontend untouched (renders `message.content`).
- `_recover_tool_call` recovers a tool call ONLY when the JSON explicitly names a tool present in `TOOL_SCHEMAS` and has an `arguments` object — NEVER guess a tool from bare args. Recovered calls run through the EXISTING execution block and respect the existing `executed` cache / `MAX_REPEATED_CALLS` / `MAX_TOOL_ITERATIONS` guards (no new unbounded loop).
- `_sanitize_answer` is conservative: only acts when the answer (stripped) starts with `{` or a ```` ``` ```` fence; normal prose (incl. prices) is returned unchanged. Never raises.
- Applied at BOTH `_run_tool_loop` return points (the no-tool-calls return and the stuck-fallback `final`). The happy-path structured `tool_calls` handling is unchanged.
- No change to Specs 1/2 behavior. No LangChain.
- Tests: `uv run pytest` (these are pure unit tests — no DB/LLM; loop tests use mock LLM + stub dispatcher like `tests/test_chat_router.py`). Lint gate: `uv run ruff format .` then `uv run ruff check .` + `uv run ruff format --check .` clean (whole repo) before each commit; no unused imports.
- Non-goals: general Markdown/HTML sanitizing; frontend; Specs 1/2.

Tool names (from `TOOL_SCHEMAS`, each item `["function"]["name"]`): `get_pricing`, `check_inventory`, `recommend_similar_products`, `prepare_order`.

---

### Task 1: Pure helpers (`_recover_tool_call`, `_sanitize_answer`) + anti-echo prompt

**Files:**
- Modify: `app/routers/chat.py` (add two module-level helpers + a `SANITIZE_FALLBACK` constant + a `_TOOL_NAMES` set)
- Modify: `app/core/prompts.py` (append anti-echo rule to `SYSTEM_PROMPT`, before the `{context}` placeholder at line 32)
- Test: `tests/test_chat_sanitize.py`

**Interfaces:**
- Consumes: `TOOL_SCHEMAS` (already imported in chat.py from `app.services.tools`).
- Produces: `SANITIZE_FALLBACK: str`; `_TOOL_NAMES: set[str]`; `_recover_tool_call(content: str, tool_names: set[str]) -> dict | None` (returns `{"name": str, "arguments": dict}` or None); `_sanitize_answer(text: str) -> str`.

- [ ] **Step 1: Write the failing tests**

`tests/test_chat_sanitize.py`:

```python
from app.routers.chat import (
    SANITIZE_FALLBACK,
    _recover_tool_call,
    _sanitize_answer,
)

_TOOLS = {"get_pricing", "check_inventory", "recommend_similar_products", "prepare_order"}


def test_recover_named_tool_call():
    got = _recover_tool_call('{"name": "get_pricing", "arguments": {"product_id": 12}}', _TOOLS)
    assert got == {"name": "get_pricing", "arguments": {"product_id": 12}}


def test_recover_nested_function_form():
    got = _recover_tool_call(
        '{"function": {"name": "check_inventory", "arguments": {"variant_id": 45}}}', _TOOLS
    )
    assert got == {"name": "check_inventory", "arguments": {"variant_id": 45}}


def test_recover_unknown_tool_name_returns_none():
    assert _recover_tool_call('{"name": "drop_table", "arguments": {}}', _TOOLS) is None


def test_recover_bare_args_no_name_returns_none():
    # the audit's leaked example — no tool name, must NOT be guessed
    assert _recover_tool_call('{"product_id": 164, "size": "M", "quantity": 1}', _TOOLS) is None


def test_recover_non_json_returns_none():
    assert _recover_tool_call("Chào bạn, mình có thể giúp gì?", _TOOLS) is None


def test_sanitize_pure_json_returns_fallback():
    assert _sanitize_answer('{"product_id": 164, "size": "M", "quantity": 1}') == SANITIZE_FALLBACK


def test_sanitize_fenced_json_only_returns_fallback():
    assert _sanitize_answer('```json\n{"name": "get_pricing"}\n```') == SANITIZE_FALLBACK


def test_sanitize_fence_with_prose_keeps_prose():
    out = _sanitize_answer('```json\n{"x":1}\n```\nDạ vợt còn hàng ạ.')
    assert "Dạ vợt còn hàng ạ." in out
    assert "{" not in out


def test_sanitize_normal_prose_unchanged():
    prose = "Quần Lining 9215 màu đen, size M còn hàng, giá 130.000đ nhé."
    assert _sanitize_answer(prose) == prose


def test_sanitize_empty_returns_fallback():
    assert _sanitize_answer("   ") == SANITIZE_FALLBACK
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_chat_sanitize.py -v`
Expected: FAIL (`cannot import name 'SANITIZE_FALLBACK'` / helpers not defined).

- [ ] **Step 3: Implement the helpers**

In `app/routers/chat.py`, add near the top (after the existing module constants like `MAX_TOOL_ITERATIONS`) and the imports (`json`, `re` — add `import re` at the top if absent):

```python
SANITIZE_FALLBACK = "Xin lỗi, bạn cho mình xin lại yêu cầu một chút nhé?"
_TOOL_NAMES: set[str] = {s["function"]["name"] for s in TOOL_SCHEMAS}


def _recover_tool_call(content: str, tool_names: set[str]) -> dict | None:
    """Parse a tool call the model emitted as JSON text in `content` (3B often
    does this instead of the structured tool_calls field). Returns
    {"name", "arguments"} ONLY when the JSON names a known tool with an args
    object; never guesses a tool from bare args. Returns None otherwise."""
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:]
        text = text.strip()
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    fn = data["function"] if isinstance(data.get("function"), dict) else data
    name = fn.get("name")
    args = fn.get("arguments")
    if name in tool_names and isinstance(args, dict):
        return {"name": name, "arguments": args}
    return None


def _sanitize_answer(text: str) -> str:
    """Strip a predominantly-JSON answer so tool-call/JSON text never reaches the
    user. Conservative: only touches answers that start with '{' or a code fence
    (normal prose, incl. prices, is returned unchanged). Empty result -> fallback."""
    stripped = text.strip()
    if not stripped:
        return SANITIZE_FALLBACK
    if not (stripped.startswith("{") or stripped.startswith("```")):
        return text
    try:
        json.loads(stripped)
        return SANITIZE_FALLBACK  # the whole answer is a JSON blob
    except (ValueError, TypeError):
        pass
    cleaned = re.sub(r"```(?:json)?.*?```", "", stripped, flags=re.DOTALL).strip()
    return cleaned if cleaned else SANITIZE_FALLBACK
```

- [ ] **Step 4: Run to verify the helper tests pass**

Run: `uv run pytest tests/test_chat_sanitize.py -v`
Expected: PASS (10 tests).

- [ ] **Step 5: Append the anti-echo rule to the system prompt**

In `app/core/prompts.py`, inside the `SYSTEM_PROMPT` triple-quoted string, in the mandatory-rules
area BEFORE the `{context}` placeholder (line 32), add this line (match the surrounding bullet
style):

```
- CHỈ trả lời khách bằng ngôn ngữ tự nhiên. TUYỆT ĐỐI KHÔNG in ra JSON, tên công cụ (tool), hay cú pháp gọi tool trong câu trả lời — đó là việc nội bộ, khách không được thấy.
```

- [ ] **Step 6: Lint + commit**

Run: `uv run ruff format app/routers/chat.py app/core/prompts.py tests/test_chat_sanitize.py && uv run ruff check . && uv run ruff format --check .`
```bash
git add app/routers/chat.py app/core/prompts.py tests/test_chat_sanitize.py
git commit -m "feat(chat): tool-call recovery + answer sanitizer helpers + anti-echo rule

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Wire the helpers into `_run_tool_loop`

**Files:**
- Modify: `app/routers/chat.py` (`_run_tool_loop`, the no-tool-calls branch ~line 188 and the stuck-fallback ~line 238)
- Test: `tests/test_chat_router.py` (extend)

**Interfaces:**
- Consumes: `_recover_tool_call`, `_sanitize_answer`, `_TOOL_NAMES` (Task 1); the existing loop machinery (`executed` cache, `MAX_REPEATED_CALLS`, `MAX_TOOL_ITERATIONS`, the `for call in tool_calls` execution block).

- [ ] **Step 1: Write the failing loop tests**

Add to `tests/test_chat_router.py` (mock LLM + the existing `_Dispatcher` stub pattern in that file):

```python
async def test_tool_loop_recovers_tool_call_from_content():
    # Turn 1: model emits a get_pricing call as JSON *content* (tool_calls empty).
    # Turn 2: model gives a natural-language answer. The loop must recover+execute
    # the call and return the clean answer — never the raw JSON.
    llm = AsyncMock()
    llm.chat_with_tools.side_effect = [
        {"role": "assistant", "content": '{"name": "get_pricing", "arguments": {"product_id": 12}}', "tool_calls": []},
        {"role": "assistant", "content": "Vợt Astrox 12 giá 1.200.000đ ạ.", "tool_calls": []},
    ]
    dispatcher = _Dispatcher({"get_pricing": json.dumps({"productId": 12, "variants": []})})
    answer, _, _ = await _run_tool_loop(llm, dispatcher, [])
    assert answer == "Vợt Astrox 12 giá 1.200.000đ ạ."
    assert dispatcher.calls == ["get_pricing"]  # recovered call was executed
    assert "{" not in answer


async def test_tool_loop_sanitizes_unrecoverable_json_content():
    # Model emits bare args (no tool name) as content and no tool_calls -> not
    # recoverable -> the answer is sanitized to the fallback, not the raw JSON.
    llm = AsyncMock()
    llm.chat_with_tools.return_value = {
        "role": "assistant",
        "content": '{"product_id": 164, "size": "M", "quantity": 1}',
        "tool_calls": [],
    }
    dispatcher = _Dispatcher({})
    answer, _, _ = await _run_tool_loop(llm, dispatcher, [])
    assert answer == SANITIZE_FALLBACK
```

Add `SANITIZE_FALLBACK` and `_run_tool_loop` to the existing imports from `app.routers.chat` at the top of the test file.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_chat_router.py -k "recovers_tool_call or sanitizes_unrecoverable" -v`
Expected: FAIL (raw JSON returned / recovery not wired).

- [ ] **Step 3: Wire the no-tool-calls branch**

In `app/routers/chat.py`, replace the `if not tool_calls:` block (currently line 188-189):

```python
        if not tool_calls:
            content = msg.get("content") or ""
            recovered = _recover_tool_call(content, _TOOL_NAMES)
            if recovered is None:
                return _sanitize_answer(content), tool_products, order_draft
            # 3B emitted the call as text — honor it: synthesize a structured call
            # and fall through to the normal execution block below.
            tool_calls = [{"function": recovered}]
            msg = {"role": "assistant", "content": "", "tool_calls": tool_calls}
```

(Execution continues at the existing `messages.append(msg)` + `for call in tool_calls:` block — the recovered call runs through the same dedup/execute path and the loop continues.)

- [ ] **Step 4: Sanitize the stuck-fallback**

In the `if repeats >= MAX_REPEATED_CALLS:` block, wrap the returned `final` (currently line 240 `return final, tool_products, order_draft`):

```python
            return _sanitize_answer(final), tool_products, order_draft
```

- [ ] **Step 5: Run the loop tests + full suite**

Run: `uv run pytest tests/test_chat_router.py -v` then `DATABASE_URL=postgresql://admin:postgresql123@localhost:5433/goodminton_test uv run pytest -q`
Expected: all pass (existing tool-loop tests unchanged in behavior + the 2 new recovery/sanitize tests).

- [ ] **Step 6: Lint + commit**

Run: `uv run ruff format app/routers/chat.py tests/test_chat_router.py && uv run ruff check . && uv run ruff format --check .`
```bash
git add app/routers/chat.py tests/test_chat_router.py
git commit -m "feat(chat): recover tool-call-as-content + sanitize answers in the tool loop

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Manual end-to-end verification (after all tasks; full stack up)

Ask the chatbot a pricing/stock question with the 3B model. Confirm: no raw JSON / `{"..."}` /
tool-call text ever appears in a reply (either the intent is recovered into a real tool call and
answered in natural language, or the reply is the polite fallback). Normal answers (with prices,
product names) render unchanged.

---

## Self-Review (against the spec)

1. **Spec coverage:** §2.1 recover → Task 1 `_recover_tool_call` + Task 2 wiring; §2.2 sanitize → Task 1 `_sanitize_answer` + Task 2 both return points; §2.3 anti-echo → Task 1 Step 5; §4 loop-safety (reuses existing guards) → Task 2 Step 3 note; §6 tests → Task 1 (helpers) + Task 2 (loop).
2. **Placeholder scan:** none — all code concrete.
3. **Type consistency:** `_recover_tool_call(content, tool_names) -> dict | None` and `_sanitize_answer(text) -> str` and `SANITIZE_FALLBACK`/`_TOOL_NAMES` identical across Task 1 defs and Task 2 uses; recovered call shape `{"function": {"name","arguments"}}` matches the existing `for call in tool_calls: fn = call.get("function", {})` reader.
