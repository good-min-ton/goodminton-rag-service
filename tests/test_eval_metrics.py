import math

from eval.metrics import (
    recall_at_k,
    mrr_at_k,
    ndcg_at_k,
    category_prf,
    bootstrap_ci,
    paired_bootstrap,
)


def test_recall_at_k_partial():
    # 2 of 3 relevant products appear in top-5
    ranked = ["a", "x", "b", "y", "z"]
    assert recall_at_k(ranked, {"a", "b", "c"}, 5) == 2 / 3


def test_recall_at_k_respects_cutoff():
    ranked = ["x", "y", "z", "a"]  # relevant 'a' is at rank 4
    assert recall_at_k(ranked, {"a"}, 3) == 0.0
    assert recall_at_k(ranked, {"a"}, 4) == 1.0


def test_mrr_at_k_first_relevant_rank():
    assert mrr_at_k(["x", "a", "b"], {"a", "b"}, 10) == 1 / 2
    assert mrr_at_k(["x", "y"], {"a"}, 10) == 0.0


def test_mrr_at_k_cutoff_excludes_late_hit():
    assert mrr_at_k(["x", "y", "z", "a"], {"a"}, 3) == 0.0


def test_ndcg_at_k_perfect_is_one():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 10) == 1.0


def test_ndcg_at_k_known_value():
    # relevant at ranks 1 and 3 → DCG = 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5
    # IDCG (2 relevant) = 1/log2(2) + 1/log2(3) = 1 + 0.6309 = 1.6309
    got = ndcg_at_k(["a", "x", "b", "y"], {"a", "b"}, 4)
    assert math.isclose(got, 1.5 / (1 + 1 / math.log2(3)), rel_tol=1e-9)


def test_category_prf_exact():
    assert category_prf({"Vợt cầu lông"}, {"Vợt cầu lông"}) == (1.0, 1.0, True)


def test_category_prf_over_prediction_penalized():
    p, r, exact = category_prf({"Vợt cầu lông", "Giày cầu lông"}, {"Vợt cầu lông"})
    assert p == 0.5 and r == 1.0 and exact is False


def test_category_prf_both_empty_is_perfect():
    assert category_prf(set(), set()) == (1.0, 1.0, True)


def test_category_prf_missed():
    assert category_prf(set(), {"Áo cầu lông"}) == (0.0, 0.0, False)


def test_bootstrap_ci_is_deterministic_and_brackets_mean():
    mean, lo, hi = bootstrap_ci([1.0, 0.0, 1.0, 1.0, 0.0], iters=500, seed=7)
    assert mean == 0.6
    assert lo <= mean <= hi
    # same seed → identical result
    assert bootstrap_ci([1.0, 0.0, 1.0, 1.0, 0.0], iters=500, seed=7) == (mean, lo, hi)


def test_paired_bootstrap_positive_delta():
    before = [0.0, 0.0, 0.0, 0.0]
    after = [1.0, 1.0, 1.0, 1.0]
    mean_delta, lo, hi = paired_bootstrap(before, after, iters=200, seed=1)
    assert mean_delta == 1.0 and lo == 1.0 and hi == 1.0
