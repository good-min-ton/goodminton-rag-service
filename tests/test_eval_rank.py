from dataclasses import dataclass
from eval.rank import rank_products


@dataclass
class _C:
    source_id: str
    chunk_index: int
    distance: float


def test_dedupe_keeps_best_rank_and_truncates():
    chunks = [
        _C("p1", 0, 0.10),
        _C("p1", 1, 0.12),  # duplicate product, worse chunk
        _C("p2", 0, 0.11),
        _C("p3", 2, 0.20),
    ]
    assert rank_products(chunks, 2) == ["p1", "p2"]


def test_tiebreak_is_deterministic_on_equal_distance():
    # equal distance → break by source_id then chunk_index
    chunks = [
        _C("pB", 0, 0.5),
        _C("pA", 3, 0.5),
        _C("pA", 1, 0.5),
    ]
    assert rank_products(chunks, 3) == ["pA", "pB"]


def test_returns_fewer_than_k_when_corpus_small():
    assert rank_products([_C("p1", 0, 0.1)], 5) == ["p1"]
