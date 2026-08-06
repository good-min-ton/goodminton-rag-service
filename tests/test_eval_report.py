from eval.fingerprint import corpus_fingerprint
from eval.report import render_markdown, render_json


def test_fingerprint_is_order_independent_and_stable():
    assert corpus_fingerprint(["2", "1", "1"]) == corpus_fingerprint(["1", "2"])
    assert corpus_fingerprint(["1", "2"]).startswith("2:")


def _agg():
    return {
        "label": "baseline",
        "corpus": {"count": 2, "fingerprint": "2:abc"},
        "k_values": [5, 10],
        "max_k": 10,
        "retrievers": {
            "production": {
                "n_queries": 3,
                "n_excluded": 1,
                "pooled": {
                    "recall@5": (0.5, 0.3, 0.7),
                    "recall@10": (0.8, 0.6, 0.9),
                    "mrr@10": (0.6, 0.4, 0.8),
                    "ndcg@10": (0.7, 0.5, 0.85),
                },
                "per_category": {"Vợt cầu lông": {"recall@10": 0.9, "n": 2}},
                "per_query_type": {"browse": {"recall@10": 0.75, "n": 3}},
                "per_source": {"hand": {"recall@10": 0.8, "n": 2}},
                "price_bucket": {"n": 1, "recall@10": 0.4},
            },
            "naive": {
                "n_queries": 3,
                "n_excluded": 1,
                "pooled": {
                    "recall@5": (0.4, 0.2, 0.6),
                    "recall@10": (0.6, 0.4, 0.8),
                    "mrr@10": (0.5, 0.3, 0.7),
                    "ndcg@10": (0.55, 0.4, 0.7),
                },
                "per_category": {},
                "per_query_type": {},
                "per_source": {},
                "price_bucket": {"n": 1, "recall@10": 0.3},
            },
        },
        "category": {
            "precision": (0.8, 0.6, 0.9),
            "recall": (0.7, 0.5, 0.85),
            "exact_match": (0.6, 0.4, 0.8),
        },
        "provenance": {"embedding_model": "bge-m3", "llm_model": "qwen2.5:14b"},
    }


def test_markdown_contains_key_sections():
    md = render_markdown(_agg())
    assert "baseline" in md and "2:abc" in md  # label + fingerprint
    assert "recall@10" in md and "0.8" in md  # pooled metric
    assert "Vợt cầu lông" in md  # per-category
    assert "browse" in md  # per-query-type
    assert "Price-constrained" in md and "0.4" in md  # price bucket
    assert "naive" in md and "production" in md  # both retrievers


def test_render_json_roundtrips_label():
    assert render_json(_agg())["label"] == "baseline"


def test_markdown_handles_nondefault_k():
    agg = {
        "label": "custom-k",
        "corpus": {"count": 1, "fingerprint": "1:def"},
        "k_values": [3, 7],
        "max_k": 7,
        "retrievers": {
            "production": {
                "n_queries": 2,
                "n_excluded": 0,
                "pooled": {
                    "recall@3": (0.4, 0.2, 0.6),
                    "recall@7": (0.7, 0.5, 0.9),
                    "mrr@7": (0.6, 0.4, 0.8),
                    "ndcg@7": (0.65, 0.5, 0.8),
                },
                "per_category": {"Vợt cầu lông": {"recall@7": 0.9, "n": 2}},
                "per_query_type": {"browse": {"recall@7": 0.75, "n": 2}},
                "per_source": {"hand": {"recall@7": 0.8, "n": 2}},
                "price_bucket": {"n": 1, "recall@7": 0.5},
            },
        },
        "category": {
            "precision": (0.8, 0.6, 0.9),
            "recall": (0.7, 0.5, 0.85),
            "exact_match": (0.6, 0.4, 0.8),
        },
        "provenance": {"embedding_model": "bge-m3", "llm_model": "qwen2.5:14b"},
    }
    md = render_markdown(agg)  # must not raise KeyError
    assert "recall@3" in md and "recall@7" in md
