# Reranker vs baseline — reranker
_Δ = cand − baseline (95% CI)_
_corpus: count=272, fingerprint=272:3ef76bf5952e_
_models: bge-m3 / qwen2.5:3b-instruct-q4_K_M_

### known-item (n=13)
| metric | baseline | cand | Δ (95% CI) |
|---|---|---|---|
| recall@10 | 0.728 | 0.758 | +0.029 [-0.066, +0.154] |
| mrr@10 | 0.689 | 0.865 | +0.176 [+0.019, +0.346] |
| ndcg@10 | 0.653 | 0.746 | +0.093 [+0.009, +0.191] |

### pooled (n=13)
| metric | baseline | cand | Δ (95% CI) |
|---|---|---|---|
| recall@10 | 0.728 | 0.758 | +0.029 [-0.066, +0.154] |
| mrr@10 | 0.689 | 0.865 | +0.176 [+0.019, +0.346] |
| ndcg@10 | 0.653 | 0.746 | +0.093 [+0.009, +0.191] |


_note: candidate_n=30 (dev CPU speed; committed default=30)_
