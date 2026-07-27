# Chatbot Tool-Result Sanitize — Design Spec

> **Scope:** Spec 3 of 3 in the sales-chatbot fix package. Closes **P4** (raw JSON / tool-call
> syntax leaking into the answer shown to the user) from
> `docs/superpowers/audits/2026-07-27-chatbot-sales-flow-audit.md`. Builds on Specs 1+2.
> **RAG-only** — the frontend just renders `message.content`, so nothing changes there.

**Date:** 2026-07-27
**Branch:** RAG `feat/chatbot-sanitize` (off `feat/chatbot-order-flow`, top of the stack). Stacked on the unmerged Foundation → Order-flow → Sanitize chain; rebase in that order.
**Repo:** `goodminton-rag-service` only.

---

## 1. Problem (P4, from the audit)

`_run_tool_loop` (`app/routers/chat.py`) detects tool calls only via the structured
`msg["tool_calls"]` field. The 3B model frequently emits a tool call as **JSON text inside
`content`** instead, leaving `tool_calls` empty. The guard `if not tool_calls: return
msg.get("content") or ""` then returns that raw JSON verbatim — e.g. `{"product_id":164,
"size":"M","quantity":1}` — straight to the user. The stuck-fallback path
(`final = await llm.chat(messages)`) likewise returns content verbatim, and the system prompt
has no rule forbidding JSON/tool-syntax in replies. Audit's expected behavior: tool calls/results
never appear in the reply; the backend parses the JSON first; the LLM's reply is natural language.

## 2. Approach — three layers of defense

Chosen scope: parser + sanitize + prompt (functional fix + guaranteed backstop + guidance).
Rejected leaner alternative (sanitize + prompt only): it stops the leak but throws away the
tool call the model was trying to make, so the customer's question goes unanswered — worse UX.

1. **Recover tool-call-as-content (functional fix).** When `tool_calls` is empty, try to parse
   `content` as a tool-call intent. Recover it ONLY when the JSON explicitly names a tool that
   exists in `TOOL_SCHEMAS` and carries an `arguments`/args object — then execute it through the
   existing tool path and continue the loop (so the 3B's intent is honored). Ambiguous JSON
   (e.g. bare args with no tool name) is NOT guessed — it falls through to sanitize.
2. **Sanitize backstop (guaranteed no leak).** `_sanitize_answer(text)` strips a leading/trailing
   JSON object or a ```` ```json ```` fence when the answer is *predominantly* JSON; if nothing
   meaningful remains, it returns a polite fallback message. Conservative — it only acts when the
   answer starts with `{`/a code fence, so normal prose (including prices like "130.000đ") is
   untouched. Applied at BOTH return points (the no-tool-calls return and the stuck-fallback).
3. **Anti-echo prompt rule.** Append to `SYSTEM_PROMPT`: reply only in natural Vietnamese, never
   print JSON, tool names, or tool-call syntax.

## 3. Components / files

- `app/routers/chat.py`:
  - New `_recover_tool_call(content: str, tool_names: set[str]) -> dict | None` — returns
    `{"name": str, "arguments": dict}` if `content` is JSON naming a known tool with an args
    object; else `None`. Accepts the common shapes: `{"name": ..., "arguments": {...}}` and
    `{"function": {"name": ..., "arguments": {...}}}`. Never guesses a tool from bare args.
  - New `_sanitize_answer(text: str) -> str` — described in §2.2; fallback message is a module
    constant `SANITIZE_FALLBACK`.
  - `_run_tool_loop`: in the `if not tool_calls:` branch, first attempt `_recover_tool_call`; on
    success synthesize a one-element `tool_calls`-shaped list and run it through the SAME
    execution block (append assistant msg, dispatch, feed tool result, continue) — respecting the
    existing dedup/`MAX_REPEATED_CALLS`/`MAX_TOOL_ITERATIONS` guards so it cannot loop forever; on
    failure `return _sanitize_answer(content), tool_products, order_draft`. Wrap the
    stuck-fallback `final` in `_sanitize_answer` too.
- `app/core/prompts.py`: append the anti-echo rule to `SYSTEM_PROMPT`.

## 4. Loop-safety

The recovery path reuses the loop's existing bounds: identical recovered calls hit the
`executed` cache → `MAX_REPEATED_CALLS` forces the tool-free final answer; the overall
`MAX_TOOL_ITERATIONS` cap still applies. A model that emits the same content-JSON every turn
therefore terminates via the existing stuck-fallback (whose output is now sanitized). No new
unbounded path.

## 5. Error handling

- `_recover_tool_call` swallows JSON parse errors → returns `None` (fall through to sanitize).
- `_sanitize_answer` never raises; worst case returns the fallback message.
- No new external calls; behavior when Ollama/tools are down is unchanged.

## 6. Testing (pytest, pure unit — no DB/LLM)

- `_recover_tool_call`: `{"name":"get_pricing","arguments":{"product_id":12}}` → dict; nested
  `{"function":{...}}` form → dict; unknown tool name → None; bare `{"product_id":164,"size":"M"}`
  (no name) → None; non-JSON prose → None.
- `_sanitize_answer`: a pure-JSON answer (starts with `{`, parses) → `SANITIZE_FALLBACK`; a
  ```` ```json {...} ``` ```` fence with no surrounding prose → `SANITIZE_FALLBACK`; a fence WITH
  surrounding prose → fence removed, prose kept; normal prose "Quần Lining 9215 giá 130.000đ..."
  (does not start with `{`/fence) → unchanged. (Conservative: only answers that START with `{` or
  a code fence are touched — prose with an embedded blob is left alone, since stripping mid-prose
  risks damaging a legitimate reply; the recovery layer + anti-echo prompt cover that rarer case.)
- `_run_tool_loop` (mock LLM + stub dispatcher, following `tests/test_chat_router.py` patterns):
  (a) LLM returns content-JSON naming `get_pricing` with empty `tool_calls` → the loop recovers +
  executes it, then a later text turn yields a clean natural-language answer (no JSON leaked);
  (b) LLM returns non-recoverable JSON as content, no tool_calls → answer is `SANITIZE_FALLBACK`;
  (c) the stuck-fallback answer is sanitized.

## 7. Non-goals

- No change to the happy-path structured `tool_calls` handling (works today).
- No change to Specs 1/2 behavior (retrieval, state machine, display_products).
- Frontend untouched (renders `message.content`).
- Not a general Markdown/HTML sanitizer — only tool-call/JSON leakage.

## 8. Open decision (proceeding with recommended default; confirm on review)

- **Three layers** (parser + sanitize + prompt), §2. The leaner two-layer option (sanitize + prompt,
  no recovery) is documented above as rejected; switching to it would drop `_recover_tool_call` and
  the recovery branch, keeping only the sanitize backstop + prompt rule.
