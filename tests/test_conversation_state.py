from app.services.conversation_state import ConversationState, ConversationStateStore


class _FakeRedis:
    """Minimal in-memory stand-in for redis.asyncio.Redis (get/set/expire)."""

    def __init__(self, raise_on=None):
        self._data: dict[str, str] = {}
        self._raise_on = raise_on  # set to "get"/"set" to simulate an outage

    async def get(self, key):
        if self._raise_on == "get":
            raise ConnectionError("redis down")
        return self._data.get(key)

    async def set(self, key, value, ex=None):
        if self._raise_on == "set":
            raise ConnectionError("redis down")
        self._data[key] = value


async def test_save_then_load_roundtrips_state():
    store = ConversationStateStore(_FakeRedis())
    state = ConversationState(
        intent="buy", categories=["pants"], price_preference="cheapest"
    )
    await store.save("sess-1", state)
    loaded = await store.load("sess-1")
    assert loaded.categories == ["pants"]
    assert loaded.price_preference == "cheapest"
    assert loaded.intent == "buy"


async def test_load_missing_session_returns_fresh_state():
    store = ConversationStateStore(_FakeRedis())
    loaded = await store.load("never-seen")
    assert loaded == ConversationState()


async def test_none_session_id_returns_fresh_and_save_is_noop():
    store = ConversationStateStore(_FakeRedis())
    await store.save(None, ConversationState(categories=["shoes"]))
    assert await store.load(None) == ConversationState()


async def test_no_client_degrades_to_stateless():
    store = ConversationStateStore(None)
    await store.save(
        "sess-1", ConversationState(categories=["pants"])
    )  # no-op, no raise
    assert await store.load("sess-1") == ConversationState()


async def test_redis_outage_degrades_without_raising():
    store = ConversationStateStore(_FakeRedis(raise_on="get"))
    assert await store.load("sess-1") == ConversationState()  # swallows ConnectionError
    store2 = ConversationStateStore(_FakeRedis(raise_on="set"))
    await store2.save("sess-1", ConversationState(categories=["pants"]))  # no raise
