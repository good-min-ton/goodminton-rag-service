"""Generic paired baseline-vs-candidate comparison over ONE golden set, in ONE
process. Groups by query_type (known-item primary), reports paired-bootstrap
delta (cand - baseline) with 95% CI. Candidate here = the reranker."""

import argparse
import asyncio
import os

import httpx

from app.core.config import settings
from app.core.db import create_pool
from app.services.embedding import EmbeddingService
from app.services.llm import LLMService
from app.services.query_understanding import QueryUnderstandingService
from app.services.retrieval import RetrievalService
from eval.fingerprint import corpus_fingerprint
from eval.golden import load_golden
from eval.metrics import mrr_at_k, ndcg_at_k, paired_bootstrap, recall_at_k
from eval.rerank_client import RerankClient
from eval.rerank_retriever import RerankRetriever
from eval.retriever import ProductionRetriever
from eval.runner import _product_source_ids

_METRICS = ("recall", "mrr", "ndcg")


def _group(rows: list[dict]) -> dict:
    g: dict = {"n": len(rows)}
    for m in _METRICS:
        b = [r["baseline"][m] for r in rows]
        c = [r["cand"][m] for r in rows]
        g[m] = {
            "baseline": sum(b) / len(b) if b else 0.0,
            "cand": sum(c) / len(c) if c else 0.0,
            "delta": paired_bootstrap(b, c),
        }
    return g


async def compare(records, baseline, cand, k: int = 10) -> dict:
    rows: list[dict] = []
    for rec in records:
        if rec.price_constrained:
            continue
        rel = set(rec.relevant_source_ids)
        entry = {"query_type": rec.query_type}
        for name, retr in (("baseline", baseline), ("cand", cand)):
            ranked = await retr.retrieve(rec.query, k)
            entry[name] = {
                "recall": recall_at_k(ranked, rel, k),
                "mrr": mrr_at_k(ranked, rel, k),
                "ndcg": ndcg_at_k(ranked, rel, k),
            }
        rows.append(entry)
    groups = {"pooled": _group(rows)}
    for qt in sorted({r["query_type"] for r in rows}):
        groups[qt] = _group([r for r in rows if r["query_type"] == qt])
    return {"k": k, "groups": groups}


def _table(title: str, g: dict, k: int) -> str:
    lines = [
        f"### {title} (n={g['n']})",
        "| metric | baseline | cand | Δ (95% CI) |",
        "|---|---|---|---|",
    ]
    for m in _METRICS:
        r = g[m]
        d, lo, hi = r["delta"]
        lines.append(
            f"| {m}@{k} | {r['baseline']:.3f} | {r['cand']:.3f} "
            f"| {d:+.3f} [{lo:+.3f}, {hi:+.3f}] |"
        )
    return "\n".join(lines)


def render_compare_markdown(result, label, corpus=None, provenance=None) -> str:
    k = result["k"]
    out = [f"# Reranker vs baseline — {label}", "_Δ = cand − baseline (95% CI)_"]
    if corpus:
        out.append(
            f"_corpus: count={corpus['count']}, fingerprint={corpus['fingerprint']}_"
        )
    if provenance:
        out.append(
            f"_models: {provenance['embedding_model']} / {provenance['llm_model']}_"
        )
    out.append("")
    groups = result["groups"]
    order = [g for g in ("known-item", "pooled") if g in groups]
    order += [g for g in sorted(groups) if g not in order]
    for name in order:
        out.append(_table(name, groups[name], k))
        out.append("")
    return "\n".join(out)


async def run_compare(
    label: str,
    golden_path: str,
    k: int,
    candidate_n: int = 30,
    query_types: set[str] | None = None,
) -> dict:
    pool = await create_pool()
    async with httpx.AsyncClient() as client:
        try:
            embedder = EmbeddingService(client)
            retrieval = RetrievalService(pool)
            qu = QueryUnderstandingService(LLMService(client))
            valid_ids = await _product_source_ids(pool)
            records, _ = load_golden(golden_path, valid_source_ids=valid_ids)
            if query_types is not None:
                records = [r for r in records if r.query_type in query_types]
            baseline = ProductionRetriever(embedder, retrieval, qu)
            reranker = RerankRetriever(
                baseline, RerankClient(client), pool, candidate_n=candidate_n
            )
            result = await compare(records, baseline, reranker, k=k)
        finally:
            await pool.close()
    corpus = {"count": len(valid_ids), "fingerprint": corpus_fingerprint(valid_ids)}
    provenance = {
        "embedding_model": settings.embedding_model,
        "llm_model": settings.llm_model,
    }
    os.makedirs("eval/reports", exist_ok=True)
    with open(f"eval/reports/compare-{label}.md", "w", encoding="utf-8") as fh:
        fh.write(render_compare_markdown(result, label, corpus, provenance))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(prog="eval.compare")
    ap.add_argument("--label", default="reranker")
    ap.add_argument("--golden", default="eval/golden.jsonl")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--candidate-n", type=int, default=30)
    ap.add_argument("--query-types", default=None)
    args = ap.parse_args()
    query_types = (
        set(args.query_types.split(",")) if args.query_types is not None else None
    )
    res = asyncio.run(
        run_compare(
            args.label,
            args.golden,
            args.k,
            candidate_n=args.candidate_n,
            query_types=query_types,
        )
    )
    print(render_compare_markdown(res, args.label))


if __name__ == "__main__":
    main()
