import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.negative_mining import (BM25, build_positive_pairs, dense_score_matrix, mine_query, sample_triplets,
                                 stable_order, verify_mining_outputs, write_jsonl)
from src.utils import install_staged

KW = dict(candidate_depth=10, skip_top=0, pool_size=3, min_score=0.0, max_score_ratio=None, flag_score_ratio=0.95)
DOCS = [f"d{i}" for i in range(8)]


# ---- BM25 (hand-computed on a toy corpus) ---------------------------------------------------
def test_bm25_matches_hand_computation():
    texts = ["apple banana", "apple apple cherry", "durian"]
    k1, b = 1.5, 0.75
    S = BM25(texts, k1=k1, b=b).score_matrix(["apple"])
    N, df, avgdl = 3, 2, (2 + 3 + 1) / 3
    idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
    def w(tf, dl): return idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
    assert S[0, 0] == pytest.approx(w(1, 2), rel=1e-5)
    assert S[0, 1] == pytest.approx(w(2, 3), rel=1e-5)
    assert S[0, 2] == 0.0


def test_bm25_unknown_query_terms_score_zero_and_case_insensitive():
    S = BM25(["Wnt signalling", "other"]).score_matrix(["WNT", "zzzz"])
    assert S[0, 0] > 0 and S[0, 1] == 0 and S[1].sum() == 0


# ---- ordering / dense --------------------------------------------------------------------------
def test_stable_order_breaks_ties_by_index():
    assert list(stable_order(np.array([1.0, 2.0, 2.0, 0.0]))) == [1, 2, 0, 3]


def test_dense_score_matrix_is_cosine_for_unit_vectors():
    a = np.array([[1, 0], [0, 1]], dtype=np.float32)
    assert np.allclose(dense_score_matrix(a, a), np.eye(2))


# ---- mine_query ----------------------------------------------------------------------------------
def row(scores): return np.array(scores, dtype=np.float32)


def test_known_positives_never_in_pool_and_pool_size_respected():
    m = mine_query("q", {"d0", "d1"}, row([9, 8, 7, 6, 5, 4, 3, 2]), DOCS, **KW)
    ids = [n["doc_id"] for n in m["negatives"]]
    assert ids == ["d2", "d3", "d4"] and m["excluded"]["known_positive"] == 2
    assert [n["rank"] for n in m["negatives"]] == [3, 4, 5]          # original ranks incl. positives


def test_best_positive_rank_and_score_reported():
    m = mine_query("q", {"d5"}, row([9, 8, 7, 6, 5, 4, 3, 2]), DOCS, **KW)
    assert m["best_positive_rank"] == 6 and m["best_positive_score"] == 4.0


def test_skip_top_and_min_score_and_depth():
    m = mine_query("q", {"d7"}, row([9, 8, 7, 0, 0, 0, 0, 1]), DOCS, **{**KW, "skip_top": 1})
    assert [n["doc_id"] for n in m["negatives"]] == ["d1", "d2"]      # d0 skipped; zero-score docs excluded
    m2 = mine_query("q", {"d7"}, row([9, 8, 7, 6, 5, 4, 3, 2]), DOCS, **{**KW, "candidate_depth": 2})
    assert [n["doc_id"] for n in m2["negatives"]] == ["d0", "d1"] and m2["n_scanned"] == 2


def test_score_ratio_flag_and_guard():
    s = row([10, 9.6, 5, 4, 3, 2, 1.5, 1.2])                           # best positive d2 -> 5
    flagged = mine_query("q", {"d2"}, s, DOCS, **KW)
    assert [n["false_negative_risk_flag"] for n in flagged["negatives"]] == [True, True, False]
    guarded = mine_query("q", {"d2"}, s, DOCS, **{**KW, "max_score_ratio": 0.95})
    assert [n["doc_id"] for n in guarded["negatives"]] == ["d3", "d4", "d5"] and guarded["excluded"]["score_ratio_guard"] == 2


def test_unlabeled_query_rejected():
    with pytest.raises(ValueError, match="no positives"):
        mine_query("q", set(), row([1] * 8), DOCS, **KW)
    with pytest.raises(ValueError, match="missing"):
        mine_query("q", {"zzz"}, row([1] * 8), DOCS, **KW)


# ---- pairs / triplets ------------------------------------------------------------------------------
ROWS = [("q2", "d3", 1), ("q1", "d1", 1), ("q1", "d0", 1), ("q1", "d0", 1)]


def test_pairs_sorted_deduped_string_ids():
    pairs = build_positive_pairs(ROWS)
    assert [p["pair_id"] for p in pairs] == ["q1__d0", "q1__d1", "q2__d3"]


def test_triplets_deterministic_and_negative_differs_from_positive():
    pairs = build_positive_pairs(ROWS)
    scores = {"q1": row([1, 2, 9, 8, 7, 6, 5, 4]), "q2": row([1, 2, 3, 0.5, 9, 8, 7, 6])}
    mined = {q: mine_query(q, {"q1": {"d0", "d1"}, "q2": {"d3"}}[q], s, DOCS, **KW) for q, s in scores.items()}
    t1, t2 = sample_triplets(pairs, mined, 42), sample_triplets(pairs, mined, 42)
    assert t1 == t2 and len(t1) == 3
    assert all(t["negative_id"] not in {"d0", "d1"} for t in t1 if t["query_id"] == "q1")
    assert len(sample_triplets(pairs, mined, 7)) == 3                  # another seed still yields one triplet per pair


# ---- gates detect corruption ----------------------------------------------------------------------
def _write_ok(tmp: Path):
    pairs = build_positive_pairs(ROWS)
    mined = [mine_query("q1", {"d0", "d1"}, row([1, 2, 9, 8, 7, 6, 5, 4]), DOCS, **KW),
             mine_query("q2", {"d3"}, row([1, 2, 3, 0.5, 9, 8, 7, 6]), DOCS, **KW)]
    write_jsonl(pairs, tmp / "train_positive_pairs.jsonl")
    write_jsonl(mined, tmp / "hard_negatives_x.jsonl")
    write_jsonl(sample_triplets(pairs, {m["query_id"]: m for m in mined}, 1), tmp / "train_triplets_x.jsonl")
    return mined


def gates(tmp, val=("v1",), test=("t1",)):
    return verify_mining_outputs(tmp, "x", ROWS, ["q1", "q2"], list(val), list(test), set(DOCS), 3)


def test_clean_outputs_pass_all_gates(tmp_path):
    _write_ok(tmp_path)
    assert all(gates(tmp_path).values())


def test_gate_catches_positive_collision(tmp_path):
    mined = _write_ok(tmp_path)
    mined[0]["negatives"][0]["doc_id"] = "d0"
    write_jsonl(mined, tmp_path / "hard_negatives_x.jsonl")
    assert gates(tmp_path)["no_known_positive_collisions"] is False


def test_gate_catches_validation_query_leak(tmp_path):
    mined = _write_ok(tmp_path)
    mined[1]["query_id"] = "v1"
    write_jsonl(mined, tmp_path / "hard_negatives_x.jsonl")
    g = gates(tmp_path)
    assert g["no_validation_or_test_queries"] is False and g["mined_queries_are_training_queries_only"] is False


def test_gate_catches_duplicates_and_unknown_docs(tmp_path):
    mined = _write_ok(tmp_path)
    mined[0]["negatives"][1]["doc_id"] = mined[0]["negatives"][0]["doc_id"]
    mined[1]["negatives"][0]["doc_id"] = "ghost"
    write_jsonl(mined, tmp_path / "hard_negatives_x.jsonl")
    g = gates(tmp_path)
    assert g["no_duplicate_negatives_within_query"] is False and g["negatives_exist_in_corpus"] is False


def test_gate_catches_bad_triplet(tmp_path):
    _write_ok(tmp_path)
    trip = [json.loads(l) for l in (tmp_path / "train_triplets_x.jsonl").read_text().splitlines()]
    trip[0]["negative_id"] = trip[0]["positive_id"]
    write_jsonl(trip, tmp_path / "train_triplets_x.jsonl")
    assert gates(tmp_path)["triplet_negative_never_its_positive"] is False


# ---- infrastructure --------------------------------------------------------------------------------
def test_install_staged_refuses_to_overwrite_changed_files(tmp_path):
    stage, dest = tmp_path / "s", tmp_path / "d"
    stage.mkdir(); dest.mkdir()
    (stage / "a.txt").write_text("new"); (dest / "a.txt").write_text("old")
    with pytest.raises(FileExistsError, match="a.txt"):
        install_staged(stage, dest)
    assert (dest / "a.txt").read_text() == "old"
    assert install_staged(stage, dest, force=True) == {"a.txt": "overwritten"}
    assert install_staged(stage, dest) == {"a.txt": "unchanged"}


def test_config_stacking_and_pinned_revision():
    from src.config import load_config
    cfg = load_config(["configs/baseline.yaml", "configs/mining.yaml"])
    assert cfg["mining"]["pool_size"] == 5 and cfg["model"]["name"].endswith("all-MiniLM-L6-v2")
    assert len(cfg["model"]["revision"]) == 40
    assert cfg["_abs"]["mining_dir"].name == "mining"


def test_samples_markdown_written_with_lf_only(tmp_path):
    """Regression: Path.write_text() produced CRLF on Windows, making regenerated files differ byte-wise."""
    from src.negative_mining import write_samples_md
    mined = [mine_query("q1", {"d0"}, row([9, 8, 7, 6, 5, 4, 3, 2]), DOCS, **KW)]
    write_samples_md(tmp_path / "s.md", mined, {"q1": "claim"}, {d: f"title {d}" for d in DOCS}, 1, 42)
    raw = (tmp_path / "s.md").read_bytes()
    assert b"\r" not in raw and raw.count(b"\n") > 3
