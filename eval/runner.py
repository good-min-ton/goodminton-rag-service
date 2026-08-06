"""Live eval runner: wire real services against dev Postgres/Ollama, aggregate,
write eval/reports/<label>.{md,json}."""

import argparse
import asyncio
import json
import os

import httpx

from app.core.config import settings
from app.core.db import create_pool
from app.services.conversation_state import ConversationState
from app.services.embedding import EmbeddingService
from app.services.llm import LLMService
from app.services.query_understanding import QueryUnderstandingService
from app.services.retrieval import RetrievalService
from eval.aggregate import category_accuracy, evaluate_retriever
from eval.fingerprint import corpus_fingerprint
from eval.golden import load_golden
from eval.report import render_markdown
from eval.retriever import NaiveVectorRetriever, ProductionRetriever

REPORTS_DIR = "eval/reports"


async def build_agg(
    records, excluded, valid_ids, retrievers, qu_analyze, label, k_values
) -> dict:
    blocks = {}
    for name, retriever in retrievers.items():
        block = await evaluate_retriever(retriever, records, k_values)
        block["n_excluded"] = len(excluded)
        blocks[name] = block
    return {
        "label": label,
        "corpus": {
            "count": len(valid_ids),
            "fingerprint": corpus_fingerprint(valid_ids),
        },
        "k_values": k_values,
        "max_k": max(k_values),
        "retrievers": blocks,
        "category": await category_accuracy(qu_analyze, records),
        "provenance": {
            "embedding_model": settings.embedding_model,
            "llm_model": settings.llm_model,
        },
    }


async def _product_source_ids(pool) -> set[str]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT DISTINCT source_id FROM kb_chunks WHERE doc_type = 'product'"
        )
    return {str(r["source_id"]) for r in rows}


async def run(label: str, k_values: list[int], golden_path: str, which: str) -> dict:
    pool = await create_pool()
    async with httpx.AsyncClient() as client:
        try:
            embedder = EmbeddingService(client)
            retrieval = RetrievalService(pool)
            qu = QueryUnderstandingService(LLMService(client))
            valid_ids = await _product_source_ids(pool)
            records, excluded = load_golden(golden_path, valid_source_ids=valid_ids)

            retrievers: dict = {}
            if which in ("naive", "both"):
                retrievers["naive"] = NaiveVectorRetriever(embedder, retrieval)
            if which in ("production", "both"):
                retrievers["production"] = ProductionRetriever(embedder, retrieval, qu)

            async def qu_analyze(query):
                return await qu.analyze(query, ConversationState())

            agg = await build_agg(
                records, excluded, valid_ids, retrievers, qu_analyze, label, k_values
            )
        finally:
            await pool.close()

    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(f"{REPORTS_DIR}/{label}.md", "w", encoding="utf-8") as fh:
        fh.write(render_markdown(agg))
    with open(f"{REPORTS_DIR}/{label}.json", "w", encoding="utf-8") as fh:
        json.dump(agg, fh, ensure_ascii=False, indent=2)
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(prog="eval")
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--k", default="5,10")
    ap.add_argument("--golden", default="eval/golden.jsonl")
    ap.add_argument(
        "--retriever", choices=["naive", "production", "both"], default="both"
    )
    args = ap.parse_args()
    k_values = sorted(int(x) for x in args.k.split(","))
    agg = asyncio.run(run(args.label, k_values, args.golden, args.retriever))
    print(
        f"Wrote {REPORTS_DIR}/{args.label}.md (corpus {agg['corpus']['fingerprint']})"
    )
