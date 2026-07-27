"""Server-side chat conversation state, keyed by session_id, stored in Redis.

Degrades to stateless (fresh state, no-op save) when Redis is absent or down —
the chat endpoint must never 500 because of state I/O.
"""

import logging

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

_KEY_PREFIX = "chat:state:"


class ConversationState(BaseModel):
    intent: str | None = None
    categories: list[str] = Field(default_factory=list)
    price_preference: str | None = None
    # RESERVED for Spec 2 (order flow) — unused in the Foundation.
    selected_product_id: int | None = None
    order_status: str = "BROWSING"
    last_confirmed_order_id: int | None = None


class ConversationStateStore:
    def __init__(self, client) -> None:
        # client: redis.asyncio.Redis | None
        self._client = client
        self._ttl = None  # set by main.py from settings.chat_state_ttl_seconds

    def with_ttl(self, ttl_seconds: int) -> "ConversationStateStore":
        self._ttl = ttl_seconds
        return self

    async def load(self, session_id: str | None) -> ConversationState:
        if not session_id or self._client is None:
            return ConversationState()
        try:
            raw = await self._client.get(_KEY_PREFIX + session_id)
        except Exception as exc:  # redis down / decode error — degrade
            log.warning("state load failed (%s); using fresh state", exc)
            return ConversationState()
        if not raw:
            return ConversationState()
        try:
            return ConversationState.model_validate_json(raw)
        except Exception as exc:
            log.warning("state parse failed (%s); using fresh state", exc)
            return ConversationState()

    async def save(self, session_id: str | None, state: ConversationState) -> None:
        if not session_id or self._client is None:
            return
        try:
            await self._client.set(
                _KEY_PREFIX + session_id,
                state.model_dump_json(),
                ex=self._ttl,
            )
        except Exception as exc:  # redis down — degrade silently
            log.warning("state save failed (%s); continuing stateless", exc)
