"""Pure, deterministic retrieval + category metrics over distinct-product ids."""

import math
import random


def recall_at_k(ranked: list[str], relevant, k: int) -> float:
    relevant = set(relevant)
    if not relevant:
        return 0.0
    hit = set(ranked[:k]) & relevant
    return len(hit) / len(relevant)


def mrr_at_k(ranked: list[str], relevant, k: int) -> float:
    relevant = set(relevant)
    for rank, sid in enumerate(ranked[:k], start=1):
        if sid in relevant:
            return 1.0 / rank
    return 0.0


def _dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 1) for i, g in enumerate(gains, start=1))


def ndcg_at_k(ranked: list[str], relevant, k: int) -> float:
    relevant = set(relevant)
    gains = [1.0 if sid in relevant else 0.0 for sid in ranked[:k]]
    idcg = _dcg([1.0] * min(len(relevant), k))
    if idcg == 0:
        return 0.0
    return _dcg(gains) / idcg


def category_prf(predicted, expected) -> tuple[float, float, bool]:
    predicted, expected = set(predicted), set(expected)
    tp = len(predicted & expected)
    precision = tp / len(predicted) if predicted else (1.0 if not expected else 0.0)
    recall = tp / len(expected) if expected else 1.0
    return (precision, recall, predicted == expected)


def bootstrap_ci(scores, iters: int = 1000, seed: int = 0):
    scores = list(scores)
    if not scores:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(scores)
    means = sorted(
        sum(scores[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters)
    )
    lo = means[int(0.025 * iters)]
    hi = means[min(int(0.975 * iters), iters - 1)]
    return (sum(scores) / n, lo, hi)


def paired_bootstrap(before, after, iters: int = 1000, seed: int = 0):
    before, after = list(before), list(after)
    if len(before) != len(after):
        raise ValueError("paired_bootstrap needs equal-length inputs")
    deltas = [b - a for a, b in zip(before, after, strict=True)]
    return bootstrap_ci(deltas, iters=iters, seed=seed)
