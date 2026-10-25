"""Phase 4 exit gates across both experiments. Status: PASS | PARTIAL (a run is missing) | FAIL.
Usage: python scripts/verify_finetune.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.utils import read_json, utc_now, write_json


def main() -> int:
    cfg = load_config(["configs/baseline.yaml", "configs/finetune.yaml"])
    reports, root = cfg["_abs"]["reports_dir"], cfg["_root"]
    exps = cfg["experiments"]
    reps, missing = {}, []
    for key, e in exps.items():
        p = reports / f"finetune_{e['run_id']}_report.json"
        (reps.__setitem__(key, read_json(p)) if p.is_file() else missing.append(key))
    checks: dict = {}
    for key, r in reps.items():
        checks[f"{key}:status_PASS"] = r["status"] == "PASS" and not r["smoke"]
        checks[f"{key}:model_dir_exists"] = (root / r["model_dir"] / "modules.json").is_file()
    if len(reps) == 2:
        a, b = reps["positive_only"], reps["hard_negative"]
        checks["same_hyperparameters"] = a["hyperparameters"] == b["hyperparameters"]
        checks["same_seed"] = a["seed"] == b["seed"]
        checks["same_base_model_revision_and_weights"] = (a["base_model"]["resolved_revision"], a["base_model"]["weights_sha256"]) == \
                                                         (b["base_model"]["resolved_revision"], b["base_model"]["weights_sha256"])
        checks["same_library_versions"] = a["library_versions"] == b["library_versions"]
        checks["same_pair_count"] = a["data"]["n_rows"] == b["data"]["n_rows"]
        checks["only_hard_negative_run_uses_negatives"] = (not a["uses_mined_negatives"]) and b["uses_mined_negatives"]
    status = "FAIL" if any(v is False for v in checks.values()) else ("PARTIAL" if missing else "PASS")
    summary = {k: {"best_stage": r["validation"]["best_stage"], "primary_metric": r["validation"]["primary_metric"],
                   "pre_training": r["validation"]["pre_training_primary"], "best": r["validation"]["best_primary"],
                   "abs_change": r["validation"]["abs_change_primary"], "rel_change_pct": r["validation"]["rel_change_primary_pct"],
                   "train_wall_s": r["timing_s"]["train_wall"], "peak_rss_mb": r["memory_mb"]["peak_rss_observed"]}
               for k, r in reps.items()}
    write_json({"status": status, "generated_utc": utc_now(), "missing_runs": missing, "checks": checks, "summary": summary,
                "note": "Evaluator-based validation numbers; Phase 5 re-scores all models with src/evaluate.py and selects ONE on validation."},
               reports / "finetune_verification.json")
    print(f"Phase 4 verification: {status} | missing: {missing} | failed: {[k for k, v in checks.items() if v is False]}")
    for k, s in summary.items():
        print(f"  {k}: {s}")
    return {"PASS": 0, "PARTIAL": 3, "FAIL": 1}[status]


if __name__ == "__main__":
    raise SystemExit(main())
