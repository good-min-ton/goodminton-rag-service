# Eval report — baseline
_corpus: count=272, fingerprint=272:3ef76bf5952e_
_models: bge-m3 / qwen2.5:3b-instruct-q4_K_M_

### naive — pooled (n=37, excluded=0)
| metric | value | 95% CI |
|---|---|---|
| recall@5 | 0.080 | [0.068, 0.092] |
| recall@10 | 0.160 | [0.138, 0.183] |
| mrr@10 | 0.897 | [0.814, 0.976] |
| ndcg@10 | 0.890 | [0.805, 0.964] |
#### Per-category (recall@10)
| key | recall@10 | n |
|---|---|---|
| Vợt cầu lông | 0.132 | 9 |
| Giày cầu lông | 0.131 | 9 |
| Quần cầu lông | 0.140 | 8 |
| Áo cầu lông | 0.142 | 8 |
| Dây cước cầu lông | 0.239 | 7 |
#### Per-query-type (recall@10)
| key | recall@10 | n |
|---|---|---|
| browse | 0.176 | 10 |
| attribute | 0.198 | 12 |
| spec | 0.191 | 6 |
| typo | 0.060 | 5 |
| multi-category | 0.090 | 4 |
#### Per-source (recall@10)
| key | recall@10 | n |
|---|---|---|
| hand | 0.157 | 29 |
| semi-auto | 0.172 | 8 |
#### Price-constrained bucket (excluded from pooled)
n=5, recall@10=0.159

### production — pooled (n=37, excluded=0)
| metric | value | 95% CI |
|---|---|---|
| recall@5 | 0.088 | [0.078, 0.097] |
| recall@10 | 0.175 | [0.156, 0.194] |
| mrr@10 | 0.982 | [0.946, 1.000] |
| ndcg@10 | 0.978 | [0.933, 1.000] |
#### Per-category (recall@10)
| key | recall@10 | n |
|---|---|---|
| Vợt cầu lông | 0.160 | 9 |
| Giày cầu lông | 0.148 | 9 |
| Quần cầu lông | 0.146 | 8 |
| Áo cầu lông | 0.146 | 8 |
| Dây cước cầu lông | 0.248 | 7 |
#### Per-query-type (recall@10)
| key | recall@10 | n |
|---|---|---|
| browse | 0.181 | 10 |
| attribute | 0.203 | 12 |
| spec | 0.191 | 6 |
| typo | 0.146 | 5 |
| multi-category | 0.090 | 4 |
#### Per-source (recall@10)
| key | recall@10 | n |
|---|---|---|
| hand | 0.175 | 29 |
| semi-auto | 0.174 | 8 |
#### Price-constrained bucket (excluded from pooled)
n=5, recall@10=0.196

### category-accuracy
- precision 0.964 [0.929, 1.000]
- recall 1.000 [1.000, 1.000]
- exact-match 0.929 [0.857, 1.000]