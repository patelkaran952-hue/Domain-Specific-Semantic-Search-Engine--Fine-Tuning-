import json
from pathlib import Path

import numpy as np
import pytest

from src.train import (TrainResult, TrainingConfigError, build_training_columns, compute_steps, extract_epoch_evals,
                       reload_fidelity, run_experiment, validate_training_config, validate_training_rows)

# Everything here is SYNTHETIC: a fake backend replaces sentence-transformers so orchestration/gates can be tested.
PRIMARY = "val_cosine_ndcg@10"


class FakeModel:
    max_len = 256


class FakeBackend:
    def __init__(self, tamper_reload=False, wrong_key=False, seq_len=256):
        self.tamper, self.wrong_key, self.seq_len = tamper_reload, wrong_key, seq_len
        self.full_eval_calls = 0
        self.loaded_saved = False

    def load_model(self, name, revision, max_seq_length):
        return FakeModel(), {"model_name": name, "resolved_revision": revision or "0" * 40, "weights_sha256": "w",
                             "embedding_dim": 4, "max_seq_length": max_seq_length}

    def encode(self, model, texts):
        base = np.array([[len(t) % 5 + 1, 1, 2, 3] for t in texts], dtype=np.float32)
        return base + (1.0 if (self.tamper and getattr(model, "reloaded", False)) else 0.0)

    def make_evaluator(self, queries, corpus, relevant, cfg):
        return {"n": len(corpus)}

    def evaluate(self, evaluator, model):
        if evaluator["n"] > 60:
            self.full_eval_calls += 1
        key = "val_other_metric" if self.wrong_key else PRIMARY
        return {key: 0.5, "val_cosine_mrr@10": 0.4}

    def primary_metric(self, evaluator, cfg):
        return PRIMARY

    def train(self, model, data, evaluator, args, run_dir):
        self.last_data, self.last_args = data, args
        hist = [{"loss": 1.0, "epoch": 0.5, "step": 5}]
        for e in range(1, args["epochs"] + 1):
            hist.append({"eval_val_cosine_ndcg@10": 0.5 + 0.1 * e, "eval_val_cosine_mrr@10": 0.4, "eval_runtime": 3.0, "epoch": float(e)})
        return TrainResult(model=model, log_history=hist, global_step=10 * args["epochs"], train_runtime_s=1.5, best_checkpoint="ck")

    def save(self, model, path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "modules.json").write_text("[]")

    def load_saved(self, path):
        m = FakeModel(); m.reloaded = True; m.max_len = self.seq_len
        return m

    def max_seq_length(self, model):
        return model.max_len

    def versions(self):
        return {"torch": "fake"}


def make_project(tmp: Path, bad: str | None = None):
    proc = tmp / "data" / "processed"; proc.mkdir(parents=True)
    docs = [{"_id": f"d{i}", "title": f"t{i}", "text": "x" * i, "doc_text": f"t{i}\n" + "x" * i} for i in range(150)]
    (proc / "corpus_processed.jsonl").write_text("\n".join(json.dumps(d) for d in docs))
    qs = [{"_id": f"q{i}", "text": f"claim {i}"} for i in range(14)]
    (proc / "queries_processed.jsonl").write_text("\n".join(json.dumps(q) for q in qs))
    train, val, test = [f"q{i}" for i in range(8)], [f"q{i}" for i in range(8, 11)], [f"q{i}" for i in range(11, 14)]
    for n, ids in (("train", train), ("val", val), ("test", test)):
        (proc / f"{n}_query_ids.json").write_text(json.dumps(ids))
    hdr = "query-id\tcorpus-id\tscore\n"
    (proc / "train_qrels.tsv").write_text(hdr + "".join(f"{q}\td{i}\t1\n" for i, q in enumerate(train)))
    (proc / "val_qrels.tsv").write_text(hdr + "".join(f"{q}\td{20 + i}\t1\n" for i, q in enumerate(val)))
    pairs = [{"pair_id": f"{q}__d{i}", "query_id": q, "positive_id": f"d{i}"} for i, q in enumerate(train)]
    trip = [{**p, "negative_id": f"d{100 + i}"} for i, p in enumerate(pairs)]
    if bad == "val_query": pairs[0]["query_id"] = "q9"
    if bad == "neg_is_pos": trip[0]["negative_id"] = trip[0]["positive_id"]
    d = tmp / "data" / "mining"; d.mkdir(parents=True)
    (d / "train_positive_pairs.jsonl").write_text("\n".join(json.dumps(p) for p in pairs))
    (d / "train_triplets_dense.jsonl").write_text("\n".join(json.dumps(p) for p in trip))
    cfg = {"_root": tmp, "_abs": {"processed_dir": proc, "reports_dir": tmp / "artifacts" / "reports", "models_dir": tmp / "artifacts" / "models"},
           "training": {"base_model": "m", "base_revision": "a" * 40, "max_seq_length": 256, "epochs": 2, "batch_size": 4, "learning_rate": 2e-5,
                        "warmup_ratio": 0.1, "weight_decay": 0.01, "seed": 42, "num_workers": 0, "batch_sampler": "no_duplicates",
                        "loss": "MultipleNegativesRankingLoss", "evaluate_on": "val"},
           "experiments": {"positive_only": {"run_id": "r_pos", "data_file": "data/mining/train_positive_pairs.jsonl", "columns": ["anchor", "positive"], "uses_mined_negatives": False},
                           "hard_negative": {"run_id": "r_hn", "data_file": "data/mining/train_triplets_dense.jsonl", "columns": ["anchor", "positive", "negative"], "uses_mined_negatives": True}}}
    return cfg


def test_full_run_passes_and_writes_artifacts(tmp_path):
    cfg, be = make_project(tmp_path), FakeBackend()
    rep = run_experiment(cfg, "hard_negative", be)
    assert rep["status"] == "PASS", {k: v for k, v in rep["gates"].items() if not v}
    run = tmp_path / "artifacts" / "models" / "r_hn"
    for f in ("training_config_resolved.json", "train_log_history.json", "val_metrics_by_epoch.json", "final/modules.json"):
        assert (run / f).is_file()
    assert (tmp_path / "artifacts/reports/finetune_r_hn_report.json").is_file()
    assert set(be.last_data) == {"anchor", "positive", "negative"} and len(be.last_data["negative"]) == 8
    assert rep["validation"]["best_stage"] == "epoch_2" and rep["validation"]["abs_change_primary"] == pytest.approx(0.2)
    assert (tmp_path / "experiments/finetune_runs.csv").read_text().count("\n") == 4   # header + pre + 2 epochs


def test_positive_only_has_no_negative_column(tmp_path):
    be = FakeBackend()
    run_experiment(make_project(tmp_path), "positive_only", be)
    assert set(be.last_data) == {"anchor", "positive"}


def test_existing_run_dir_never_overwritten(tmp_path):
    cfg = make_project(tmp_path)
    run_experiment(cfg, "positive_only", FakeBackend())
    with pytest.raises(SystemExit, match="never overwritten"):
        run_experiment(cfg, "positive_only", FakeBackend())


def test_pretrained_evaluation_is_cached_across_runs(tmp_path):
    cfg, be = make_project(tmp_path), FakeBackend()
    # force "full" corpus size > 60 docs for the counter: project has 150 docs
    run_experiment(cfg, "positive_only", be)
    run_experiment(cfg, "hard_negative", be)
    assert be.full_eval_calls == 1


def test_smoke_mode_writes_outside_reports_and_skips_runs_csv(tmp_path):
    cfg = make_project(tmp_path)
    rep = run_experiment(cfg, "positive_only", FakeBackend(), smoke=True)
    assert rep["smoke"] and (tmp_path / "artifacts/smoke/finetune_r_pos_report.json").is_file()
    assert not (tmp_path / "artifacts/reports/finetune_r_pos_report.json").exists()
    assert not (tmp_path / "experiments/finetune_runs.csv").exists()


def test_validation_query_in_training_data_rejected(tmp_path):
    with pytest.raises(TrainingConfigError, match="gates failed"):
        run_experiment(make_project(tmp_path, bad="val_query"), "positive_only", FakeBackend())


def test_negative_equal_to_positive_rejected(tmp_path):
    with pytest.raises(TrainingConfigError, match="gates failed"):
        run_experiment(make_project(tmp_path, bad="neg_is_pos"), "hard_negative", FakeBackend())


def test_preflight_catches_wrong_metric_key_before_training(tmp_path):
    be = FakeBackend(wrong_key=True)
    with pytest.raises(TrainingConfigError, match="primary metric"):
        run_experiment(make_project(tmp_path), "positive_only", be)
    assert not hasattr(be, "last_data")                       # training never started


def test_reload_mismatch_and_seq_len_loss_fail_gates(tmp_path):
    rep = run_experiment(make_project(tmp_path), "positive_only", FakeBackend(tamper_reload=True, seq_len=128))
    assert rep["status"] == "FAIL"
    assert rep["gates"]["reloaded_equals_in_memory"] is False and rep["gates"]["max_seq_length_preserved"] is False


# ---- config + helpers ------------------------------------------------------------------------------------
@pytest.mark.parametrize("mut,msg", [
    (lambda c: c["experiments"]["positive_only"].update(columns=["anchor"]), "columns"),
    (lambda c: c["experiments"]["positive_only"].update(uses_mined_negatives=True), "consume"),
    (lambda c: c["training"].update(batch_size=512), "batch_size"),
    (lambda c: c["training"].update(epochs=20), "epochs"),
    (lambda c: c["training"].update(evaluate_on="test"), "test.tsv"),
    (lambda c: c["training"].update(batch_sampler="batch_sampler"), "no_duplicates"),
    (lambda c: c["training"].update(num_workers=2), "Windows"),
    (lambda c: c["training"].update(base_revision="abc"), "40-char"),
])
def test_config_validation(tmp_path, mut, msg):
    cfg = make_project(tmp_path)
    mut(cfg)
    with pytest.raises(TrainingConfigError, match=msg):
        validate_training_config(cfg, "positive_only")


def test_real_finetune_yaml_is_valid_and_comparable():
    from src.config import find_repo_root, load_config
    cfg = load_config(["configs/baseline.yaml", "configs/finetune.yaml"], root=find_repo_root())
    validate_training_config(cfg, "positive_only"); validate_training_config(cfg, "hard_negative")
    assert cfg["training"]["base_revision"] == cfg["model"]["revision"]
    assert cfg["training"]["max_seq_length"] == cfg["model"]["max_seq_length"]


def test_compute_steps_and_extract_evals_and_fidelity():
    assert compute_steps(742, 8, 2, 0.1) == {"steps_per_epoch": 93, "total_steps": 186, "warmup_steps": 19}
    ev = extract_epoch_evals([{"loss": 1, "epoch": 0.5}, {"eval_val_cosine_ndcg@10": 0.6, "eval_runtime": 2, "eval_loss": 1, "epoch": 1.0}])
    assert ev == {1.0: {"val_cosine_ndcg@10": 0.6}}
    a = np.ones((2, 3), dtype=np.float32)
    assert reload_fidelity(a, a, 1e-5)["within_tol"] and not reload_fidelity(a, a + 1, 1e-5)["within_tol"]
    assert not reload_fidelity(a, np.ones((3, 3)), 1e-5)["shapes_match"]
    assert not reload_fidelity(a, np.full((2, 3), np.nan), 1e-5)["all_finite"]


def test_build_training_columns_maps_ids_to_text():
    cols = build_training_columns([{"query_id": "q", "positive_id": "p", "negative_id": "n"}], ["anchor", "positive", "negative"],
                                  {"q": "QTEXT"}, {"p": "PTEXT", "n": "NTEXT"})
    assert cols == {"anchor": ["QTEXT"], "positive": ["PTEXT"], "negative": ["NTEXT"]}


# ---- failure recording / incomplete-run handling ------------------------------------------------------------------
class ExplodingBackend(FakeBackend):
    def train(self, *a, **k):
        raise RuntimeError("boom: simulated trainer failure")


def test_failure_is_recorded_with_traceback_and_log(tmp_path):
    cfg = make_project(tmp_path)
    with pytest.raises(RuntimeError, match="boom"):
        run_experiment(cfg, "positive_only", ExplodingBackend())
    run = tmp_path / "artifacts" / "models" / "r_pos"
    d = json.loads((run / "run_failed.json").read_text())
    assert d["stage"] == "train" and d["exception_type"] == "RuntimeError" and "boom" in d["traceback"]
    assert (run / "train.log").is_file()
    assert not (tmp_path / "artifacts/reports/finetune_r_pos_report.json").exists()


def test_incomplete_run_blocks_rerun_unless_archived_then_is_renamed_not_deleted(tmp_path):
    cfg = make_project(tmp_path)
    with pytest.raises(RuntimeError):
        run_experiment(cfg, "positive_only", ExplodingBackend())
    with pytest.raises(SystemExit, match="did not finish"):
        run_experiment(cfg, "positive_only", FakeBackend())
    rep = run_experiment(cfg, "positive_only", FakeBackend(), archive_incomplete=True)
    assert rep["status"] == "PASS"
    archived = list((tmp_path / "artifacts" / "models").glob("r_pos.incomplete-*"))
    assert len(archived) == 1 and (archived[0] / "run_failed.json").is_file()      # evidence preserved


def test_finished_run_is_never_archived_even_with_flag(tmp_path):
    cfg = make_project(tmp_path)
    run_experiment(cfg, "positive_only", FakeBackend())
    with pytest.raises(SystemExit, match="FINISHED"):
        run_experiment(cfg, "positive_only", FakeBackend(), archive_incomplete=True)


def test_diagnose_describes_failed_and_finished_runs(tmp_path):
    from src.diagnose import describe_pretrained_cache, describe_run
    cfg = make_project(tmp_path)
    with pytest.raises(RuntimeError):
        run_experiment(cfg, "positive_only", ExplodingBackend())
    run_experiment(cfg, "hard_negative", FakeBackend())
    models, reports = tmp_path / "artifacts/models", tmp_path / "artifacts/reports"
    bad = "\n".join(describe_run(models / "r_pos", reports / "finetune_r_pos_report.json"))
    good = "\n".join(describe_run(models / "r_hn", reports / "finetune_r_hn_report.json"))
    assert "MISSING" in bad and "boom" in bad and "ABSENT" in bad
    assert "FOUND" in good and "evaluation entries" in good
    assert "PRETRAINED" in "\n".join(describe_pretrained_cache(reports / "finetune_pretrained_val_evaluator.json"))
    assert describe_run(tmp_path / "nope", tmp_path / "x")[-1].strip() == "(does not exist)"
