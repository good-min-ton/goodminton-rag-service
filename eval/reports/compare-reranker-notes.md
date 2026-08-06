# Reranker vs baseline — interpretation & caveats

_Companion to `compare-reranker.md` (corpus 272, fingerprint `272:3ef76bf5952e`, models bge-m3 / qwen2.5:3b, reranker = bge-reranker-v2-m3 cross-encoder)._

## Conclusion

**The cross-encoder reranker IMPROVES within-set ranking on the known-item slice —
statistically significant on nDCG@10.** This is the positive complement to the
negative B2-Phase-1 hybrid (trgm) result: lexical fusion did not help, semantic
cross-encoder re-scoring does.

Confirmed at **candidate_n=30** (the committed default, run on CPU over the known-item
slice — the GPU was not needed):

| metric (known-item, n=13, candidate_n=30) | baseline | reranker | Δ (95% CI) |
|---|---|---|---|
| recall@10 | 0.728 | 0.758 | +0.029 [−0.066, +0.154] (positive, ns) |
| MRR@10 | 0.689 | 0.865 | +0.176 [+0.019, +0.346] ← CI clear of 0 |
| **nDCG@10** | 0.653 | 0.746 | **+0.093 [+0.009, +0.191]** ← CI clear of 0 |

The nDCG@10 gain (+0.093) matches the earlier candidate_n=10 run (+0.091) → **not a
candidate_n artifact**; at candidate_n=30 the MRR gain also reaches significance and
recall@10 now moves positively (the reranker can pull a relevant product from ranks
11–30 into the top-10), though recall is not yet significant at n=13.

**Recommendation:** the reranker is worth adopting into the production retrieval path
(`app/routers/chat.py`) — a separate "adopt" step. The known-item gain is confirmed at
the default candidate_n=30; a full-55-golden run would need GPU/more time but the
category slices are near-ceiling (baseline nDCG ~1.0), so add little.

## Why it wins (vs hybrid trgm losing)

The cross-encoder reads the (query, product) pair jointly and scores semantic
relevance — so it correctly promotes the product a Vietnamese query describes, even
when surface tokens differ. trgm only matched characters and mostly added noise.

## Caveats (read before citing)

- **Slice scope: known-item only (13 q) on CPU.** bge-reranker-v2-m3 (~600M) on CPU is
  slow, so the full 55-query golden isn't run here; the known-item slice (the primary
  readout) is. The category slices are near-ceiling (baseline nDCG ~1.0) so the reranker
  adds little there. Reproduce exactly (reranker service running):
  `EMBED_SERVICE_URL=http://localhost:8002 uv run python -m eval.compare --label reranker --k 10 --candidate-n 30 --query-types known-item`
- **candidate_n=30 = committed default** — the earlier candidate_n=10 run gave the same
  nDCG@10 gain (+0.091 vs +0.093), confirming it is not a candidate_n artifact.
- **Upper-bound for the slice.** known-item queries are anchored to features in each
  product's own indexed text (documented in the golden). The reranker winning here is
  encouraging but should be re-confirmed on held-out, real-register queries before
  strong claims.
- **n=13** — CI is wide; treat significance as indicative.
