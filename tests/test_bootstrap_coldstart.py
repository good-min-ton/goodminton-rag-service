"""Cold-start bootstrap must heal a half-populated knowledge base.

The old guard was `product_count == 0`: a backfill that died halfway (Ollama
restarting, shop-api not healthy yet) left some chunks behind, so every later
`up -d` logged "present -> skip" and the missing products stayed missing. A
question about one of them then retrieved whatever else happened to be indexed.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import backfill_products  # noqa: E402
import bootstrap  # noqa: E402


class _FakeConn:
    def __init__(self, static_count: int, indexed: int):
        self._static = static_count
        self._indexed = indexed
        self.closed = False

    async def fetchval(self, sql, *args):
        return self._static if "'static'" in sql else self._indexed

    async def close(self):
        self.closed = True


@pytest.fixture
def wiring(monkeypatch):
    """Stub every side effect; return the call log the assertions read."""
    calls: dict[str, object] = {"indexed_ids": None, "static": False}

    monkeypatch.setattr(
        backfill_products, "resolve_database_url", lambda: "postgresql://x/y"
    )

    async def fake_index_ids(ids):
        calls["indexed_ids"] = list(ids)
        return calls.get("failures", 0)

    async def fake_static_main():
        calls["static"] = True

    monkeypatch.setattr(backfill_products, "index_ids", fake_index_ids)
    monkeypatch.setattr(bootstrap.index_static_docs, "main", fake_static_main)
    return calls


def _use_db(monkeypatch, *, static_count, indexed, missing):
    conn = _FakeConn(static_count, indexed)

    async def fake_wait_for_db(dsn):
        return conn

    async def fake_missing(_conn):
        return missing

    monkeypatch.setattr(bootstrap, "wait_for_db", fake_wait_for_db)
    monkeypatch.setattr(backfill_products, "fetch_unindexed_product_ids", fake_missing)
    return conn


@pytest.mark.asyncio
async def test_indexes_only_the_products_that_are_missing(monkeypatch, wiring):
    """The regression this exists for: chunks are present, so the old guard
    skipped — but three products have none."""
    _use_db(monkeypatch, static_count=12, indexed=197, missing=[5, 8, 13])

    failed = await bootstrap.main()

    assert wiring["indexed_ids"] == [5, 8, 13]
    assert wiring["static"] is False  # static docs already present
    assert failed == 0


@pytest.mark.asyncio
async def test_skips_when_every_visible_product_is_indexed(monkeypatch, wiring):
    _use_db(monkeypatch, static_count=12, indexed=200, missing=[])

    assert await bootstrap.main() == 0
    assert wiring["indexed_ids"] is None


@pytest.mark.asyncio
async def test_empty_store_indexes_static_docs_and_every_product(monkeypatch, wiring):
    _use_db(monkeypatch, static_count=0, indexed=0, missing=[1, 2, 3])

    await bootstrap.main()

    assert wiring["static"] is True
    assert wiring["indexed_ids"] == [1, 2, 3]


@pytest.mark.asyncio
async def test_failures_are_returned_so_rag_init_can_exit_non_zero(monkeypatch, wiring):
    """A one-shot container that exits 0 reports success no matter how much it
    dropped, which is how a partial cold start went unnoticed."""
    wiring["failures"] = 2
    _use_db(monkeypatch, static_count=12, indexed=100, missing=[7, 9])

    assert await bootstrap.main() == 2


@pytest.mark.asyncio
async def test_db_connection_is_closed_even_when_a_query_raises(monkeypatch, wiring):
    conn = _use_db(monkeypatch, static_count=12, indexed=1, missing=[])

    async def boom(_conn):
        raise RuntimeError("query failed")

    monkeypatch.setattr(backfill_products, "fetch_unindexed_product_ids", boom)

    with pytest.raises(RuntimeError):
        await bootstrap.main()
    assert conn.closed is True
