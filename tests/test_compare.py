import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

import src.compare as C
from src.compare import (Loaded, ProtocolError, bootstrap_mean_ci, fingerprint, ledger_append, ledger_assert_unused, ledger_entries,
                         ledger_key, paired_bootstrap, run_test_stage, run_validation_stage, select_final)
from src.index_faiss import IndexAlignmentError
from src.utils import sha256_file, write_json

# Everything below is SYNTHETIC (hashing "encoders", tiny corpus): it tests protocol/mechanics, not model quality.
KEYS = ["baseline", "ft_pos_only", "ft_hardneg_dense"]
DIMS = {"baseline": 24, "ft_pos_only": 48, "ft_hardneg_dense": 96}
N_DOCS = 60


def hash_encoder(dim):
    def fn(batch):
        out = np.zeros((len(batch), dim), dtype=np.float32)
        for i, t in enumerate(batch):
            for w in t.lower().split():
                out[i, zlib.crc32(w.encode()) % dim] += 1.0
        return out
    return fn


def make_loader(weight_override=None):
    def loader(key, cand):
        wsha = (weight_override or {}).get(key, f"w_{key}")
        prov = {"model_name": key, "resolved_revision": f"rev_{key}", "weights_sha256": wsha, "max_seq_length": 256, "embedding_dim": DIMS[key]}
        return Loaded(hash_encoder(DIMS[key]), prov)
    return loader


def make_cfg(tmp: Path, test_hash_ok=True, p4_ok=True):
    proc, rep = tmp / "data/processed", tmp / "artifacts/reports"
    for d in (proc, rep, tmp / "data/raw", tmp / "artifacts/embeddings", tmp / "artifacts/indexes"):
        d.mkdir(parents=True, exist_ok=True)
    docs = [{"_id": f"d{i:03d}", "title": f"title {i}", "text": f"abstract {i}", "doc_text": f"topic{i} alpha{i % 5} beta{i % 7} gamma{i % 3}"} for i in range(N_DOCS)]
    (proc / "corpus_processed.jsonl").write_text("\n".join(json.dumps(d) for d in docs))
    write_json([d["_id"] for d in docs], proc / "document_ids.json")
    split = {"train": range(0, 8), "val": range(8, 14), "test": range(14, 22)}
    qs = [{"_id": f"q{i:02d}", "text": f"topic{i + 10} alpha{(i + 10) % 5} noise{i}"} for i in range(22)]   # positive = doc i+10
    (proc / "queries_processed.jsonl").write_text("\n".join(json.dumps(q) for q in qs))
    hdr = "query-id\tcorpus-id\tscore\n"
    for name, r in split.items():
        write_json([f"q{i:02d}" for i in r], proc / f"{name}_query_ids.json")
    (proc / "val_qrels.tsv").write_text(hdr + "".join(f"q{i:02d}\td{i + 10:03d}\t1\n" for i in split["val"]))
    (tmp / "data/raw/test.tsv").write_text(hdr + "".join(f"q{i:02d}\td{i + 10:03d}\t1\n" for i in split["test"]))
    write_json({"raw_file_sha256": {"test_qrels": sha256_file(tmp / "data/raw/test.tsv") if test_hash_ok else "0" * 64}}, rep / "data_audit.json")
    for k in KEYS[1:]:                                              # Phase 4 artifacts
        md = tmp / "artifacts/models" / k / "final"; md.mkdir(parents=True, exist_ok=True)
        (md / "model.safetensors").write_bytes(k.encode())
        write_json({"status": "PASS", "model_weights_sha256": sha256_file(md / "model.safetensors"),
                    "validation": {"best_primary": 0.5}}, rep / f"finetune_{k}_report.json")
    write_json({"status": "PASS" if p4_ok else "FAIL"}, rep / "finetune_verification.json")
    return {"_root": tmp, "_abs": {"processed_dir": proc, "reports_dir": rep, "embeddings_dir": tmp / "artifacts/embeddings",
                                   "indexes_dir": tmp / "artifacts/indexes"},
            "_raw": {"test_qrels": tmp / "data/raw/test.tsv"}, "project": {"seed": 42}, "model": {"max_seq_length": 256},
            "embedding": {"batch_size": 8, "normalize": True, "dtype": "float32", "chunk_size": 32}, "index": {"type": "IndexFlatIP"},
            "retrieval": {"top_k": 20}, "evaluation": {"ks": [1, 5, 10], "ndcg_k": 10, "mrr_k": 10, "precision_k": 10, "map_k": 10},
            "checks": {"norm_tolerance": 1e-4, "score_tolerance": 1e-5, "n_consistency_samples": 5, "reproducibility_resample_docs": 8,
                       "reproducibility_tolerance": 1e-4},
            "candidates": {"baseline": {"kind": "pretrained", "label": "B"},
                           "ft_pos_only": {"kind": "local", "label": "P", "run_id": "ft_pos_only", "model_dir": "artifacts/models/ft_pos_only/final"},
                           "ft_hardneg_dense": {"kind": "local", "label": "H", "run_id": "ft_hardneg_dense", "model_dir": "artifacts/models/ft_hardneg_dense/final"}},
            "selection": {"split": "val", "metric": "ndcg@10", "rule": "argmax_point_estimate", "tie_break_order": KEYS},
            "bootstrap": {"n_resamples": 500, "seed": 42, "confidence": 0.95, "metrics": ["ndcg@10", "recall@10", "mrr@10", "recall@1"]},
            "comparisons": [["ft_pos_only", "baseline"], ["ft_hardneg_dense", "baseline"], ["ft_hardneg_dense", "ft_pos_only"]],
            "failure_cases": {"n_cases": 3, "snippet_chars": 60}, "gates": {"evaluator_agreement_tol": 1.0, "baseline_report_tol": 1e-6},
            "latency": {"n_warmup": 1}}


# ---- statistics / selection / ledger ------------------------------------------------------------------------------------
def test_bootstrap_is_seeded_and_ci_brackets_mean():
    v = np.random.default_rng(0).uniform(size=50)
    a, b = bootstrap_mean_ci(v, 1000, 1, 0.95), bootstrap_mean_ci(v, 1000, 1, 0.95)
    assert a == b and a["ci_low"] <= a["mean"] <= a["ci_high"]


def test_paired_bootstrap_detects_consistent_gain_and_null():
    base = {f"q{i}": 0.5 for i in range(40)}
    better = {q: 0.6 for q in base}
    r = paired_bootstrap(better, base, 1000, 1, 0.95)
    assert r["mean_diff"] == pytest.approx(0.1) and r["ci_excludes_zero"] and r["queries_better"] == 40 and r["queries_worse"] == 0
    rng = np.random.default_rng(1)
    noisy = {q: 0.5 + rng.choice([-0.1, 0.1]) for q in base}
    assert not paired_bootstrap(noisy, base, 2000, 1, 0.95)["ci_excludes_zero"]
    with pytest.raises(ValueError):
        paired_bootstrap({"a": 1}, {"b": 1}, 10, 1, 0.95)


def test_select_final_argmax_and_tie_break_prefers_simplest():
    assert select_final({"baseline": 0.6, "ft_pos_only": 0.7, "ft_hardneg_dense": 0.65}, KEYS) == "ft_pos_only"
    assert select_final({"baseline": 0.7, "ft_pos_only": 0.7, "ft_hardneg_dense": 0.7}, KEYS) == "baseline"
    with pytest.raises(ValueError):
        select_final({"x": 1.0}, KEYS)


def test_ledger_blocks_repeat(tmp_path):
    p = tmp_path / "ledger.json"
    k = ledger_key("m", "w", "fp")
    ledger_assert_unused(p, [k])
    ledger_append(p, [{"entry_key": k}])
    with pytest.raises(ProtocolError, match="already evaluated"):
        ledger_assert_unused(p, [k])
    ledger_assert_unused(p, [ledger_key("m", "w2", "fp")])          # different weights => different frozen config


def test_fingerprint_changes_with_any_input():
    assert fingerprint({"a": 1}) == fingerprint({"a": 1}) != fingerprint({"a": 2})


# ---- validation stage ------------------------------------------------------------------------------------------------------
def test_validation_stage_end_to_end_freezes_argmax_and_never_reads_test(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    real = C.read_qrels
    def guarded(path):
        assert Path(path).name != "test.tsv", "validation stage must not read test.tsv"
        return real(path)
    monkeypatch.setattr(C, "read_qrels", guarded)
    rep = run_validation_stage(cfg, make_loader())
    assert rep["status"] == "PASS", {k: v for k, v in rep["gates"].items() if v is not True}
    sel = json.loads((tmp_path / "artifacts/reports/model_selection.json").read_text())
    scores = {k: rep["models"][k]["metrics"]["ndcg@10"] for k in KEYS}
    assert sel["status"] == "FROZEN" and scores[sel["selected"]] == max(scores.values())
    assert rep["eval_setup"]["n_queries"] == 6 and rep["eval_setup"]["candidate_corpus_size"] == N_DOCS
    for k in KEYS:
        assert (tmp_path / f"artifacts/reports/val_per_query_{k}.csv").is_file()
        assert (tmp_path / f"artifacts/indexes/{k}_flatip.faiss").is_file()
    assert (tmp_path / f"artifacts/reports/val_failure_cases_{sel['selected']}.md").is_file()
    runs = (tmp_path / "experiments/runs.csv").read_text().splitlines()
    assert len(runs) == 4 and all(",val," in r for r in runs[1:])
    assert "paired_comparisons" in rep and "ft_pos_only__minus__baseline" in rep["paired_comparisons"]


def test_validation_not_frozen_when_a_gate_fails(tmp_path):
    cfg = make_cfg(tmp_path, p4_ok=False)
    rep = run_validation_stage(cfg, make_loader())
    assert rep["status"] == "FAIL" and rep["gates"]["phase4_verification_PASS"] is False
    assert json.loads((tmp_path / "artifacts/reports/model_selection.json").read_text())["status"] == "NOT_FROZEN"


def test_baseline_must_reproduce_phase2_report(tmp_path):
    cfg = make_cfg(tmp_path)
    write_json({"metrics": {"ndcg@10": 0.123456, "recall@1": 0.1}}, tmp_path / "artifacts/reports/baseline_val_metrics.json")
    rep = run_validation_stage(cfg, make_loader())
    assert rep["gates"]["baseline:matches_phase2_report"] is False and rep["status"] == "FAIL"


def test_validation_refuses_overwrite_without_force_and_after_test(tmp_path):
    cfg = make_cfg(tmp_path)
    run_validation_stage(cfg, make_loader())
    with pytest.raises(SystemExit, match="--force"):
        run_validation_stage(cfg, make_loader())
    cfg2 = make_cfg(tmp_path)
    run_validation_stage(cfg2, make_loader(), force=True)
    run_test_stage(cfg2, make_loader())
    with pytest.raises(ProtocolError, match="test-guided"):
        run_validation_stage(cfg2, make_loader(), force=True)


# ---- test stage -------------------------------------------------------------------------------------------------------------
def frozen_project(tmp_path, **kw):
    cfg = make_cfg(tmp_path, **kw)
    run_validation_stage(cfg, make_loader())
    return cfg


def test_test_stage_runs_once_on_frozen_models_only(tmp_path):
    cfg = frozen_project(tmp_path)
    sel = json.loads((tmp_path / "artifacts/reports/model_selection.json").read_text())
    rep = run_test_stage(cfg, make_loader())
    assert rep["status"] == "PASS", {k: v for k, v in rep["gates"].items() if v is not True}
    assert set(rep["models"]) == set(sel["test_candidates"]) and rep["eval_setup"]["n_queries"] == 8
    ledger = ledger_entries(tmp_path / "artifacts/reports/test_evaluation_ledger.json")
    assert {e["model"] for e in ledger} == set(sel["test_candidates"]) and all("metrics" in e for e in ledger)
    assert (tmp_path / "artifacts/reports/test_results.json").is_file()
    assert ",test," in (tmp_path / "experiments/runs.csv").read_text()
    with pytest.raises(ProtocolError, match="already evaluated"):
        run_test_stage(cfg, make_loader())


def test_test_stage_requires_frozen_selection(tmp_path):
    with pytest.raises(ProtocolError, match="validation"):
        run_test_stage(make_cfg(tmp_path), make_loader())
    cfg = make_cfg(tmp_path / "b", p4_ok=False)
    run_validation_stage(cfg, make_loader())
    with pytest.raises(ProtocolError, match="FROZEN"):
        run_test_stage(cfg, make_loader())


def test_test_stage_blocked_by_changed_weights(tmp_path):
    cfg = frozen_project(tmp_path)
    sel = json.loads((tmp_path / "artifacts/reports/model_selection.json").read_text())
    with pytest.raises(ProtocolError, match="weights differ"):
        run_test_stage(cfg, make_loader({sel["selected"]: "tampered"}))
    assert not (tmp_path / "artifacts/reports/test_evaluation_ledger.json").exists()      # nothing consumed


def test_test_stage_blocked_by_changed_config(tmp_path):
    cfg = frozen_project(tmp_path)
    cfg["retrieval"]["top_k"] = 10
    with pytest.raises(ProtocolError, match="fingerprint"):
        run_test_stage(cfg, make_loader())


def test_test_stage_blocked_if_raw_test_file_changed(tmp_path):
    cfg = frozen_project(tmp_path, test_hash_ok=False)
    with pytest.raises(ProtocolError, match="immutable"):
        run_test_stage(cfg, make_loader())


def test_existing_mismatched_index_is_never_overwritten(tmp_path):
    cfg = make_cfg(tmp_path)
    run_validation_stage(cfg, make_loader())
    emb = tmp_path / "artifacts/embeddings"
    for f in emb.glob("ft_pos_only_docs.*"):
        f.unlink()                                                  # force re-encode with a DIFFERENT encoder dim
    DIMS["ft_pos_only"] = 40
    try:
        with pytest.raises(IndexAlignmentError, match="refusing to overwrite"):
            run_validation_stage(cfg, make_loader(), force=True)
    finally:
        DIMS["ft_pos_only"] = 48


def test_failure_case_markdown_has_sections():
    pq = {k: {"q1": {"ndcg@10": v, "recall@10": 0.0 if v == 0 else 1.0}} for k, v in (("a", 0.0), ("b", 0.5))}
    scored = {k: {"q1": [("d1", 0.9), ("d2", 0.8)]} for k in pq}
    md = C.failure_cases_markdown("T", "a", "b", pq, scored, {"q1": {"d9"}}, {"d1": "t1", "d2": "t2", "d9": "t9"}, {"q1": "claim"}, 3, 50, 10, "note")
    assert "No judged positive in the top 10" in md and "Largest regressions" in md and "claim" in md and "note" in md


def _load_cli():
    import importlib.util
    spec = importlib.util.spec_from_file_location("evaluate_models_cli", Path(__file__).resolve().parents[1] / "scripts" / "evaluate_models.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_cli_validation_then_test_prints_tables_and_requires_confirmation(tmp_path, monkeypatch, capsys):
    cli, cfg = _load_cli(), make_cfg(tmp_path)
    monkeypatch.setattr(cli, "load_config", lambda *_a, **_k: cfg)
    monkeypatch.setattr(cli, "real_loader", lambda _cfg: make_loader())
    monkeypatch.setattr(sys, "argv", ["evaluate_models.py", "--stage", "validation"])
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "VALIDATION" in out and "SELECTED" in out and "ft_pos_only__minus__baseline" in out
    monkeypatch.setattr(sys, "argv", ["evaluate_models.py", "--stage", "test"])
    assert cli.main() == 2 and not (tmp_path / "artifacts/reports/test_results.json").exists()      # no confirmation -> nothing runs
    monkeypatch.setattr(sys, "argv", ["evaluate_models.py", "--stage", "test", "--confirm-single-test-evaluation"])
    assert cli.main() == 0 and "HELD-OUT TEST" in capsys.readouterr().out
    assert cli.main() == 4 and "PROTOCOL BLOCK" in capsys.readouterr().out                         # second attempt is blocked
