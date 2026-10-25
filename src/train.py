"""Phase 4 orchestration: validate config + training data, run one fine-tuning experiment through a backend,
apply exit gates, and write logs/reports. All library-version-sensitive calls live in src/train_backend.py so
that everything here is unit-testable without torch / sentence-transformers.

Rules enforced here: training rows come from the TRAINING partition only; validation queries are used only for
evaluation / best-checkpoint selection; test.tsv is never read; existing run directories are never overwritten.
"""
from __future__ import annotations

import csv
import json
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import psutil

from .utils import get_logger, read_json, sha256_file, utc_now, write_json

log = get_logger("train")

ALLOWED_COLUMNS = (["anchor", "positive"], ["anchor", "positive", "negative"])
METRIC_SUFFIXES = ["ndcg@10", "mrr@10", "map@10", "recall@1", "recall@5", "recall@10",
                   "precision@10", "accuracy@1", "accuracy@5", "accuracy@10"]


class TrainingConfigError(Exception):
    pass


@dataclass
class TrainResult:
    model: Any                                   # best model (load_best_model_at_end) after training
    log_history: list[dict]
    global_step: int
    train_runtime_s: float
    best_checkpoint: str | None = None
    extra: dict = field(default_factory=dict)


class Backend(Protocol):
    def load_model(self, name: str, revision: str | None, max_seq_length: int) -> tuple[Any, dict]: ...
    def encode(self, model: Any, texts: list[str]) -> np.ndarray: ...
    def make_evaluator(self, queries: dict, corpus: dict, relevant: dict, cfg: dict) -> Any: ...
    def evaluate(self, evaluator: Any, model: Any) -> dict[str, float]: ...
    def primary_metric(self, evaluator: Any, cfg: dict) -> str: ...
    def train(self, model: Any, data: dict[str, list[str]], evaluator: Any, args: dict, run_dir: Path) -> TrainResult: ...
    def save(self, model: Any, path: Path) -> None: ...
    def load_saved(self, path: Path) -> Any: ...
    def max_seq_length(self, model: Any) -> int: ...
    def versions(self) -> dict: ...


# ------------------------------------------------------------------------------------ validation
def validate_training_config(cfg: dict, exp_key: str) -> dict:
    t, exps = cfg.get("training"), cfg.get("experiments", {})
    if not t or exp_key not in exps:
        raise TrainingConfigError(f"configs/finetune.yaml must define `training` and experiments.{exp_key}")
    e = exps[exp_key]
    errs = []
    if e["columns"] not in ALLOWED_COLUMNS:
        errs.append(f"columns must be one of {ALLOWED_COLUMNS}, got {e['columns']}")
    if bool(e.get("uses_mined_negatives")) != ("negative" in e["columns"]):
        errs.append("uses_mined_negatives must be true iff the 'negative' column is used (the loss must actually consume negatives)")
    if t.get("loss") != "MultipleNegativesRankingLoss":
        errs.append("only MultipleNegativesRankingLoss is supported (it consumes explicit negatives as a 3rd column)")
    if t.get("evaluate_on") != "val":
        errs.append("training.evaluate_on must be 'val' (test.tsv must never drive training or selection)")
    if t.get("batch_sampler") != "no_duplicates":
        errs.append("batch_sampler must be 'no_duplicates' to avoid false in-batch negatives")
    if not (1 <= int(t["epochs"]) <= 5):
        errs.append("epochs must be in 1..5 for this CPU-oriented project")
    if not (1 <= int(t["batch_size"]) <= 64):
        errs.append("batch_size must be in 1..64 (8 GB RAM, CPU)")
    if not float(t["learning_rate"]) > 0:
        errs.append("learning_rate must be > 0")
    if t.get("base_revision") is not None and len(str(t["base_revision"])) != 40:
        errs.append("base_revision should be a 40-char commit hash (or null)")
    if int(t.get("num_workers", 0)) != 0:
        errs.append("num_workers must be 0 on Windows")
    if errs:
        raise TrainingConfigError("Invalid training config:\n  " + "\n  ".join(errs))
    return e


def validate_training_rows(rows: list[dict], columns: list[str], train_ids, val_ids, test_ids, train_pos: set[tuple[str, str]],
                           queries: dict, doc_text: dict) -> dict[str, bool]:
    """Raise TrainingConfigError on any structural problem; return gate booleans otherwise."""
    if not rows:
        raise TrainingConfigError("training data file is empty")
    need = {"query_id", "positive_id"} | ({"negative_id"} if "negative" in columns else set())
    bad = [i for i, r in enumerate(rows) if not need <= set(r)]
    if bad:
        raise TrainingConfigError(f"rows {bad[:5]} lack required keys {sorted(need)}")
    tr, va, te = set(train_ids), set(val_ids), set(test_ids)
    qs = {r["query_id"] for r in rows}
    unknown = [r for r in rows if r["query_id"] not in queries or r["positive_id"] not in doc_text
               or ("negative_id" in r and r["negative_id"] not in doc_text)]
    if unknown:
        raise TrainingConfigError(f"{len(unknown)} rows reference IDs missing from processed queries/corpus, e.g. {unknown[0]}")
    gates = {
        "queries_are_training_queries_only": qs <= tr,
        "no_validation_or_test_queries": not (qs & (va | te)),
        "pairs_are_judged_train_positives": all((r["query_id"], r["positive_id"]) in train_pos for r in rows),
        "negative_never_a_known_positive": all(
            (r["query_id"], r["negative_id"]) not in train_pos for r in rows if "negative_id" in r),
        "no_duplicate_rows": len({tuple(sorted(r.items())) for r in rows}) == len(rows),
    }
    return gates


def build_training_columns(rows: list[dict], columns: list[str], queries: dict, doc_text: dict) -> dict[str, list[str]]:
    src = {"anchor": ("query_id", queries), "positive": ("positive_id", doc_text), "negative": ("negative_id", doc_text)}
    return {c: [src[c][1][r[src[c][0]]] for r in rows] for c in columns}


# ------------------------------------------------------------------------------------ helpers
def compute_steps(n_rows: int, batch_size: int, epochs: int, warmup_ratio: float) -> dict[str, int]:
    per_epoch = math.ceil(n_rows / batch_size)
    total = per_epoch * epochs
    return {"steps_per_epoch": per_epoch, "total_steps": total, "warmup_steps": int(math.ceil(warmup_ratio * total))}


def extract_epoch_evals(log_history: list[dict]) -> dict[float, dict[str, float]]:
    """Per-epoch validation metrics from trainer.state.log_history (evaluator keys, 'eval_' prefix stripped)."""
    skip = ("runtime", "samples_per_second", "steps_per_second", "loss")
    out: dict[float, dict[str, float]] = {}
    for entry in log_history:
        ev = {k[5:]: float(v) for k, v in entry.items() if k.startswith("eval_") and not any(s in k for s in skip)
              and isinstance(v, (int, float))}
        if ev and "epoch" in entry:
            out[round(float(entry["epoch"]), 4)] = ev
    return out


def reload_fidelity(a: np.ndarray, b: np.ndarray, tol: float) -> dict[str, Any]:
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    ok_shape = a.shape == b.shape and a.ndim == 2
    diff = float(np.max(np.abs(a - b))) if ok_shape else float("inf")
    return {"shapes_match": ok_shape, "all_finite": bool(np.isfinite(b).all()), "max_abs_diff": diff, "within_tol": diff <= tol}


class RssMonitor:
    """Samples process RSS in a background thread to report an observed peak (not a precise profile)."""

    def __init__(self, interval_s: float = 2.0):
        self.interval, self.peak_mb, self._stop = interval_s, 0.0, threading.Event()
        self._p, self._t = psutil.Process(), threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, self._p.memory_info().rss / 1e6)
            self._stop.wait(self.interval)

    def __enter__(self):
        self._t.start(); return self

    def __exit__(self, *a):
        self._stop.set(); self._t.join(timeout=3)
        self.peak_mb = max(self.peak_mb, self._p.memory_info().rss / 1e6)


def append_runs_csv(path: Path, run_id: str, stages: dict[str, dict], model_rev: str) -> None:
    cols = ["timestamp_utc", "run_id", "stage", "epoch", "evaluator", *METRIC_SUFFIXES]
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(cols)
        for stage, (epoch, metrics) in stages.items():
            vals = [next((f"{v:.6f}" for k, v in metrics.items() if k.endswith(f"cosine_{s}")), "") for s in METRIC_SUFFIXES]
            w.writerow([utc_now(), run_id, stage, epoch, "sentence-transformers InformationRetrievalEvaluator", *vals])


# ------------------------------------------------------------------------------------ main entry
def run_experiment(cfg: dict, exp_key: str, backend: Backend, smoke: bool = False) -> dict:
    root: Path = cfg["_root"]
    exp, t = validate_training_config(cfg, exp_key), cfg["training"]
    proc, reports = cfg["_abs"]["processed_dir"], cfg["_abs"]["reports_dir"]
    models_dir = (cfg["_abs"]["models_dir"] if not smoke else root / "artifacts" / "smoke" / "finetune")
    out_reports = reports if not smoke else root / "artifacts" / "smoke"
    run_dir = models_dir / exp["run_id"]
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"{run_dir} already exists and is not empty. Existing models are never overwritten: "
                         "move or delete it yourself if you really want to retrain.")

    # ---- inputs -------------------------------------------------------------------------------
    from .data_io import read_jsonl, read_qrels

    docs = read_jsonl(proc / "corpus_processed.jsonl")
    doc_text = {d["_id"]: d["doc_text"] for d in docs}
    queries = {r["_id"]: r["text"] for r in read_jsonl(proc / "queries_processed.jsonl")}
    train_ids, val_ids, test_ids = (read_json(proc / f"{n}_query_ids.json") for n in ("train", "val", "test"))
    train_pos = {(q, d) for q, d, s in read_qrels(proc / "train_qrels.tsv") if s > 0}
    val_rel: dict[str, set[str]] = {}
    for q, d, s in read_qrels(proc / "val_qrels.tsv"):          # validation judgments: evaluation only
        if s > 0:
            val_rel.setdefault(q, set()).add(d)
    data_path = root / exp["data_file"]
    if not data_path.is_file():
        raise SystemExit(f"Missing training data {data_path}. Run scripts/mine_negatives.py (Phase 3) first.")
    rows = [json.loads(l) for l in data_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    data_gates = validate_training_rows(rows, exp["columns"], train_ids, val_ids, test_ids, train_pos, queries, doc_text)
    if not all(data_gates.values()):
        raise TrainingConfigError(f"training-data gates failed: {data_gates}")
    if smoke:
        rows = rows[:32]
    data = build_training_columns(rows, exp["columns"], queries, doc_text)

    # ---- evaluator corpora ----------------------------------------------------------------------
    full_q = {q: queries[q] for q in sorted(val_rel)}
    if smoke:
        keep = sorted(val_rel)[:16]
        small_docs = sorted({d for q in keep for d in val_rel[q]} | set(list(doc_text)[:100]))
        eval_q, eval_c, eval_r = {q: queries[q] for q in keep}, {d: doc_text[d] for d in small_docs}, {q: val_rel[q] for q in keep}
    else:
        eval_q, eval_c, eval_r = full_q, doc_text, {q: val_rel[q] for q in full_q}

    # ---- model, preflight, pre-training evaluation ------------------------------------------------
    from .utils import set_seed

    set_seed(t["seed"])
    model, prov = backend.load_model(t["base_model"], t["base_revision"], t["max_seq_length"])
    if t["base_revision"] and prov["resolved_revision"] != t["base_revision"]:
        raise TrainingConfigError(f"resolved revision {prov['resolved_revision']} != configured {t['base_revision']}")
    # tiny preflight: fails fast if evaluator keys / API differ from expectations, before any long computation
    pre_keep = sorted(val_rel)[:8]
    pre_docs = sorted({d for q in pre_keep for d in val_rel[q]} | set(list(doc_text)[:40]))
    pre_ev = backend.make_evaluator({q: queries[q] for q in pre_keep}, {d: doc_text[d] for d in pre_docs},
                                    {q: val_rel[q] for q in pre_keep}, cfg)
    pre_res = backend.evaluate(pre_ev, model)
    primary = backend.primary_metric(pre_ev, cfg)
    if primary not in pre_res:
        raise TrainingConfigError(f"Evaluator primary metric {primary!r} not in results {sorted(pre_res)[:12]}...")
    log.info("Preflight OK; primary metric = %s", primary)

    evaluator = backend.make_evaluator(eval_q, eval_c, eval_r, cfg)
    cache_p = out_reports / "finetune_pretrained_val_evaluator.json"
    ckey = {"base_revision": prov["resolved_revision"], "weights_sha256": prov["weights_sha256"],
            "corpus_sha256": sha256_file(proc / "corpus_processed.jsonl"), "val_qrels_sha256": sha256_file(proc / "val_qrels.tsv"),
            "max_seq_length": t["max_seq_length"], "n_queries": len(eval_q), "n_docs": len(eval_c)}
    if cache_p.is_file() and read_json(cache_p).get("key") == ckey:
        pre_metrics = read_json(cache_p)["metrics"]
        log.info("Reusing cached pre-training evaluator metrics (%s)", cache_p.name)
    else:
        t0 = time.perf_counter()
        pre_metrics = backend.evaluate(evaluator, model)
        write_json({"key": ckey, "metrics": pre_metrics, "seconds": round(time.perf_counter() - t0, 1),
                    "note": "Same evaluator as during training: an apples-to-apples epoch-0 reference."}, cache_p)

    # ---- training ------------------------------------------------------------------------------------
    steps = compute_steps(len(rows), t["batch_size"], 1 if smoke else t["epochs"], t["warmup_ratio"])
    args = {"epochs": 1 if smoke else t["epochs"], "batch_size": t["batch_size"], "learning_rate": float(t["learning_rate"]),
            "warmup_steps": steps["warmup_steps"], "weight_decay": float(t["weight_decay"]), "seed": t["seed"],
            "num_workers": 0, "batch_sampler": "no_duplicates", "metric_for_best_model": f"eval_{primary}",
            "logging_steps": 10}
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json({"experiment": exp_key, "experiment_cfg": exp, "training_cfg": t, "trainer_args": args, "steps": steps,
                "provenance": prov, "data_file": exp["data_file"], "data_sha256": sha256_file(data_path), "n_rows": len(rows),
                "smoke": smoke}, run_dir / "training_config_resolved.json")
    t0 = time.perf_counter()
    with RssMonitor() as mon:
        res = backend.train(model, data, evaluator, args, run_dir)
    wall = time.perf_counter() - t0
    write_json(res.log_history, run_dir / "train_log_history.json")

    # ---- save + reload gates --------------------------------------------------------------------------
    final = run_dir / "final"
    backend.save(res.model, final)
    probe = [doc_text[d] for d in list(doc_text)[:4]]
    in_mem = backend.encode(res.model, probe)
    reloaded = backend.load_saved(final)
    fid = reload_fidelity(in_mem, backend.encode(reloaded, probe), 1e-4)
    evals = extract_epoch_evals(res.log_history)
    stages = {"pre_training": (0, pre_metrics), **{f"epoch_{int(e) if float(e).is_integer() else e}": (e, m) for e, m in sorted(evals.items())}}
    best_stage = max((s for s in stages if s != "pre_training"), key=lambda s: stages[s][1].get(primary, -1), default=None)
    gates = {**{f"data:{k}": v for k, v in data_gates.items()},
             "training_logged": bool(res.log_history) and res.global_step > 0,
             "validation_evaluated_each_epoch": len(evals) >= args["epochs"],
             "validation_metric_present_every_stage": all(primary in m for _, m in stages.values()),
             "model_saved": (final / "modules.json").is_file(),
             "reloaded_model_embeds_finite_correct_shape": fid["shapes_match"] and fid["all_finite"],
             "reloaded_equals_in_memory": fid["within_tol"],
             "max_seq_length_preserved": backend.max_seq_length(reloaded) == t["max_seq_length"],
             "embedding_dim_preserved": int(backend.encode(reloaded, probe[:1]).shape[1]) == int(prov["embedding_dim"]),
             "test_split_untouched_by_construction": True}
    weights = next((p for p in (final / "model.safetensors", final / "0_Transformer" / "model.safetensors") if p.exists()), None)
    pre_v, best_v = pre_metrics.get(primary), (stages[best_stage][1].get(primary) if best_stage else None)
    report = {
        "run_id": exp["run_id"], "experiment": exp_key, "status": "PASS" if all(gates.values()) else "FAIL",
        "smoke": smoke, "generated_utc": utc_now(), "seed": t["seed"], "gates": gates, "reload_check": fid,
        "uses_mined_negatives": exp["uses_mined_negatives"], "columns": exp["columns"],
        "data": {"file": exp["data_file"], "sha256": sha256_file(data_path), "n_rows": len(rows)},
        "hyperparameters": {**{k: t[k] for k in ("epochs", "batch_size", "learning_rate", "warmup_ratio", "weight_decay",
                                                 "max_seq_length", "batch_sampler", "loss")}, **steps},
        "base_model": prov, "library_versions": backend.versions(),
        "validation": {"evaluator": "sentence-transformers InformationRetrievalEvaluator (name=val, cosine)",
                       "primary_metric": primary, "by_stage": {s: m for s, (_, m) in stages.items()},
                       "best_stage": best_stage, "pre_training_primary": pre_v, "best_primary": best_v,
                       "abs_change_primary": None if None in (pre_v, best_v) else best_v - pre_v,
                       "rel_change_primary_pct": None if not pre_v or best_v is None else 100.0 * (best_v - pre_v) / pre_v,
                       "caveat": "Evaluator metric definitions can differ slightly from src/evaluate.py; Phase 5 re-scores every "
                                 "model with the shared implementation. Negative changes are valid results."},
        "timing_s": {"train_wall": round(wall, 1), "trainer_reported_train_runtime": res.train_runtime_s,
                     "global_step": res.global_step},
        "memory_mb": {"peak_rss_observed": round(mon.peak_mb, 1), "note": "sampled every 2s; not a precise profile"},
        "model_dir": str(final.relative_to(root)) if final.is_relative_to(root) else str(final),
        "model_weights_sha256": sha256_file(weights) if weights else None,
        "best_checkpoint": res.best_checkpoint,
        "caveats": ["Mined negatives (if used) are unjudged and may include false negatives.",
                    "Validation drives best-epoch selection within a run; test.tsv was not used."],
    }
    write_json(report, out_reports / f"finetune_{exp['run_id']}_report.json")
    write_json({s: {"epoch": e, "metrics": m} for s, (e, m) in stages.items()}, run_dir / "val_metrics_by_epoch.json")
    if not smoke:
        append_runs_csv(root / "experiments" / "finetune_runs.csv", exp["run_id"], stages, prov["resolved_revision"])
    return report
