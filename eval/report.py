"""Render the aggregated eval result to markdown + JSON."""


def _ci(t) -> str:
    m, lo, hi = t
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def _pooled_table(name: str, r: dict, k_values: list, max_k: int) -> str:
    lines = [
        f"### {name} — pooled (n={r['n_queries']}, excluded={r['n_excluded']})",
        "| metric | value | 95% CI |",
        "|---|---|---|",
    ]
    keys = [f"recall@{k}" for k in k_values] + [f"mrr@{max_k}", f"ndcg@{max_k}"]
    for key in keys:
        m, lo, hi = r["pooled"][key]
        lines.append(f"| {key} | {m:.3f} | [{lo:.3f}, {hi:.3f}] |")
    return "\n".join(lines)


def _slice_table(title: str, slyce: dict, max_k: int) -> str:
    if not slyce:
        return ""
    metric = f"recall@{max_k}"
    lines = [f"#### {title}", f"| key | {metric} | n |", "|---|---|---|"]
    for key, v in slyce.items():
        lines.append(f"| {key} | {v[metric]:.3f} | {v['n']} |")
    return "\n".join(lines)


def render_markdown(agg: dict) -> str:
    c = agg["corpus"]
    k_values = agg["k_values"]
    max_k = agg["max_k"]
    out = [
        f"# Eval report — {agg['label']}",
        f"_corpus: count={c['count']}, fingerprint={c['fingerprint']}_",
        f"_models: {agg['provenance']['embedding_model']} / "
        f"{agg['provenance']['llm_model']}_",
        "",
    ]
    for name, r in agg["retrievers"].items():
        out.append(_pooled_table(name, r, k_values, max_k))
        out.append(
            _slice_table(f"Per-category (recall@{max_k})", r["per_category"], max_k)
        )
        out.append(
            _slice_table(f"Per-query-type (recall@{max_k})", r["per_query_type"], max_k)
        )
        out.append(_slice_table(f"Per-source (recall@{max_k})", r["per_source"], max_k))
        pb = r["price_bucket"]
        out.append(
            f"#### Price-constrained bucket (excluded from pooled)\n"
            f"n={pb['n']}, recall@{max_k}={pb[f'recall@{max_k}']:.3f}"
        )
        out.append("")
    cat = agg["category"]
    out.append("### category-accuracy")
    out.append(
        f"- precision {_ci(cat['precision'])}\n"
        f"- recall {_ci(cat['recall'])}\n"
        f"- exact-match {_ci(cat['exact_match'])}"
    )
    return "\n".join(out)


def render_json(agg: dict) -> dict:
    return agg
