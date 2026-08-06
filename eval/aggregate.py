"""Retriever-agnostic aggregation: per-query metrics → pooled + slices + CIs."""

from collections import defaultdict

from eval.metrics import (
    bootstrap_ci,
    category_prf,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


async def evaluate_retriever(retriever, records, k_values: list[int]) -> dict:
    max_k = max(k_values)
    scored, price_recall = [], []
    for rec in records:
        ranked = await retriever.retrieve(rec.query, max_k)
        rel = set(rec.relevant_source_ids)
        row = {
            "rec": rec,
            **{f"recall@{k}": recall_at_k(ranked, rel, k) for k in k_values},
            f"mrr@{max_k}": mrr_at_k(ranked, rel, max_k),
            f"ndcg@{max_k}": ndcg_at_k(ranked, rel, max_k),
        }
        if rec.price_constrained:
            price_recall.append(row[f"recall@{max_k}"])
        else:
            scored.append(row)

    metric_keys = [f"recall@{k}" for k in k_values] + [
        f"mrr@{max_k}",
        f"ndcg@{max_k}",
    ]
    pooled = {mk: bootstrap_ci([r[mk] for r in scored]) for mk in metric_keys}

    def _slice(attr: str) -> dict:
        buckets: dict = defaultdict(list)
        for r in scored:
            key = getattr(r["rec"], attr)
            buckets[key].append(r[f"recall@{max_k}"])
        return {
            k: {f"recall@{max_k}": _mean(v), "n": len(v)} for k, v in buckets.items()
        }

    per_category: dict = defaultdict(list)
    for r in scored:
        for cat in r["rec"].expected_categories:
            per_category[cat].append(r[f"recall@{max_k}"])

    return {
        "n_queries": len(scored),
        "n_excluded": 0,  # runner overwrites with corpus-drift excludes
        "pooled": pooled,
        "per_category": {
            k: {f"recall@{max_k}": _mean(v), "n": len(v)}
            for k, v in per_category.items()
        },
        "per_query_type": _slice("query_type"),
        "per_source": _slice("source"),
        "price_bucket": {
            "n": len(price_recall),
            f"recall@{max_k}": _mean(price_recall),
        },
    }


async def category_accuracy(qu_analyze, records) -> dict:
    precisions, recalls, exacts = [], [], []
    for rec in records:
        understanding = await qu_analyze(rec.query)
        p, r, exact = category_prf(
            set(understanding.categories), set(rec.expected_categories)
        )
        precisions.append(p)
        recalls.append(r)
        exacts.append(1.0 if exact else 0.0)
    return {
        "precision": bootstrap_ci(precisions),
        "recall": bootstrap_ci(recalls),
        "exact_match": bootstrap_ci(exacts),
    }
