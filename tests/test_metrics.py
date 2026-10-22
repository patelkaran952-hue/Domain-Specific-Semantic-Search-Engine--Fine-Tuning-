import math

import pytest

from src.evaluate import evaluate_run, qrels_to_dict, query_metrics

pytest.importorskip("numpy")


def test_single_positive_at_rank_2():
    m = query_metrics(["x", "a", "y"], {"a"})
    assert m["recall@1"] == 0 and m["recall@5"] == 1 and m["recall@10"] == 1
    assert m["mrr@10"] == pytest.approx(0.5)
    assert m["ndcg@10"] == pytest.approx(1 / math.log2(3))
    assert m["precision@10"] == pytest.approx(0.1)
    assert m["map@10"] == pytest.approx(0.5)


def test_two_positives_hand_computed():
    m = query_metrics(["a", "x", "b"], {"a", "b"})
    assert m["recall@1"] == pytest.approx(0.5) and m["recall@5"] == 1
    assert m["mrr@10"] == 1
    assert m["ndcg@10"] == pytest.approx((1 + 0.5) / (1 + 1 / math.log2(3)))
    assert m["map@10"] == pytest.approx((1 + 2 / 3) / 2)


def test_perfect_ranking_scores_one():
    m = query_metrics(["a", "b", "z"], {"a", "b"})
    assert m["ndcg@10"] == pytest.approx(1.0) and m["map@10"] == pytest.approx(1.0)


def test_positive_beyond_cutoff_scores_zero():
    ranked = [f"d{i}" for i in range(10)] + ["a"]
    m = query_metrics(ranked, {"a"})
    assert m["recall@10"] == 0 and m["mrr@10"] == 0 and m["ndcg@10"] == 0 and m["map@10"] == 0


def test_duplicate_docs_in_ranking_not_double_counted():
    m = query_metrics(["a", "a", "a"], {"a", "b"})
    assert m["recall@10"] == pytest.approx(0.5)


def test_more_positives_than_k_ideal_dcg_capped():
    pos = {f"p{i}" for i in range(15)}
    m = query_metrics([f"p{i}" for i in range(10)], pos)
    assert m["ndcg@10"] == pytest.approx(1.0) and m["map@10"] == pytest.approx(1.0)


def test_requires_positives():
    with pytest.raises(ValueError):
        query_metrics(["a"], set())


def test_aggregate_includes_queries_missing_from_run_as_zero():
    qrels = {"q1": {"a"}, "q2": {"b"}}
    res = evaluate_run({"q1": ["a"]}, qrels)
    assert res["n_queries"] == 2 and res["queries_missing_from_run"] == ["q2"]
    assert res["aggregate"]["recall@1"] == pytest.approx(0.5)
    assert res["aggregate"]["ndcg@10"] == pytest.approx(0.5)


def test_run_queries_without_labels_ignored_but_reported():
    res = evaluate_run({"q1": ["a"], "zzz": ["a"]}, {"q1": {"a"}})
    assert res["run_queries_without_labels_ignored"] == ["zzz"] and res["n_queries"] == 1


def test_qrels_to_dict_keeps_string_ids_and_groups():
    assert qrels_to_dict([("007", "1", 1), ("007", "2", 1), ("8", "1", 0)]) == {"007": {"1", "2"}}
