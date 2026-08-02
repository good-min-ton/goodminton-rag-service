"""Offline evaluation harness for the RAG chatbot.

Measures retrieval quality and the reranker's lift on a labeled question set
(eval/dataset.jsonl). Run on the server (needs DB + Ollama):

    docker compose -f docker-compose.prod.yml exec -T rag-service \\
        uv run python scripts/eval_chatbot.py            # retrieval + rerank
    ... scripts/eval_chatbot.py --no-rerank              # retrieval only (fast)

Labels (per line):
  type = product | policy | out_of_scope
  expect_name_any = [substrings]  -> a product whose name contains any counts as hit
  expect_category = "..."         -> a product in this category counts as hit
Metrics:
  - product: recall@k, MRR; hit@N cosine vs hit@N reranked (rerank lift)
  - policy: fraction retrieving a static (doc_type='static') chunk
  - out_of_scope: fraction where nearest product is beyond card_max_distance
"""

import argparse
import asyncio
import json
import os
import sys
import unicodedata

import asyncpg
import httpx
from pgvector.asyncpg import register_vector

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.config import settings  # noqa: E402
from app.services.embedding import EmbeddingService  # noqa: E402
from app.services.llm import LLMService  # noqa: E402
from app.services.rerank import RerankService  # noqa: E402
from app.services.retrieval import RetrievalService  # noqa: E402
from backfill_products import resolve_database_url  # noqa: E402

DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval", "dataset.jsonl"
)


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def _name(chunk) -> str:
    first = chunk.content.split("\n", 1)[0]
    p = "Sản phẩm:"
    return first[len(p) :].strip() if first.startswith(p) else first


def _hit(chunk, item) -> bool:
    if chunk.doc_type != "product":
        return False
    name = _norm(_name(chunk))
    for sub in item.get("expect_name_any", []):
        if _norm(sub) in name:
            return True
    return False  # category handled separately (needs metadata, best-effort by name)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--k", type=int, default=settings.retrieval_candidates)
    args = ap.parse_args()

    items = [json.loads(line) for line in open(DATA, encoding="utf-8") if line.strip()]
    pool = await asyncpg.create_pool(
        dsn=resolve_database_url(), init=lambda c: register_vector(c)
    )
    async with httpx.AsyncClient() as http:
        embed = EmbeddingService(http)
        retr = RetrievalService(pool)
        rer = RerankService(LLMService(http), http)

        prod = [
            i for i in items if i["type"] == "product" and (i.get("expect_name_any"))
        ]
        pol = [i for i in items if i["type"] == "policy"]
        oos = [i for i in items if i["type"] == "out_of_scope"]

        recall_k = mrr = cos_topn = rr_topn = 0
        n_named = len(prod)
        topn = settings.chat_display_products_max
        print(
            f"\n{'ID':<5}{'type':<7}{'r@k':<5}{'rank':<6}"
            f"{'cos@N':<7}{'rr@N':<6} question"
        )
        print("-" * 78)
        for it in prod:
            vec = await embed.embed(it["question"])
            chunks = await retr.search(vec, k=args.k)
            prods = [c for c in chunks if c.doc_type == "product"]
            hits = [j for j, c in enumerate(prods) if _hit(c, it)]
            in_k = bool(hits)
            rank = (hits[0] + 1) if hits else 0
            recall_k += in_k
            mrr += (1 / rank) if rank else 0
            cos_hit = any(_hit(c, it) for c in prods[:topn])
            cos_topn += cos_hit
            rr_hit = cos_hit
            if not args.no_rerank and prods:
                cand = [
                    {"id": c.source_id, "name": _name(c), "text": c.content[:300]}
                    for c in prods
                ]
                ranked = await rer.rerank(it["question"], cand, topn)
                idset = set(ranked)
                rr_hit = any(c.source_id in idset and _hit(c, it) for c in prods)
            rr_topn += rr_hit
            print(
                f"{it['id']:<5}{'prod':<7}{('Y' if in_k else '.'):<5}"
                f"{(rank or '-'):<6}{('Y' if cos_hit else '.'):<7}"
                f"{('Y' if rr_hit else '.'):<6}{it['question'][:34]}"
            )

        for it in pol:
            vec = await embed.embed(it["question"])
            chunks = await retr.search(vec, k=args.k)
            ok = any(c.doc_type == "static" for c in chunks[:topn])
            print(f"{it['id']:<5}{'pol':<7}{('Y' if ok else '.'):<5}")
        for it in oos:
            vec = await embed.embed(it["question"])
            chunks = await retr.search(vec, k=args.k)
            prods = [c for c in chunks if c.doc_type == "product"]
            weak = (not prods) or min(
                c.distance for c in prods
            ) > settings.card_max_distance
            print(
                f"{it['id']:<5}{'oos':<7}{('Y' if weak else '.'):<5} "
                f"(nearest={min((c.distance for c in prods), default=9):.3f})"
            )

        print("-" * 78)
        if n_named:
            print(
                f"Product ({n_named} câu):  recall@{args.k} = "
                f"{recall_k / n_named:.2f}   MRR = {mrr / n_named:.2f}"
            )
            print(
                f"Card hit@{topn}:  cosine = {cos_topn / n_named:.2f}"
                + ("" if args.no_rerank else f"   reranked = {rr_topn / n_named:.2f}")
            )
    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
