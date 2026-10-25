"""Phase 4: fine-tune MiniLM on TRAINING-partition data; evaluate on VALIDATION queries only (test.tsv never read).

Usage (PowerShell, repo root, env active; needs: python -m pip install "sentence-transformers[train]"):
    python scripts/finetune.py --experiment positive_only --smoke     # 32 rows, 1 epoch, tiny eval corpus (pipeline check)
    python scripts/finetune.py --experiment positive_only             # run 1: (anchor, positive) pairs
    python scripts/finetune.py --experiment hard_negative             # run 2: (anchor, positive, mined negative) triplets
An existing model directory is NEVER overwritten; the script stops and tells you what to move/delete.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.train import run_experiment


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True, choices=["positive_only", "hard_negative"])
    ap.add_argument("--config", nargs="+", default=["configs/baseline.yaml", "configs/finetune.yaml"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--archive-incomplete", action="store_true",
                    help="if an earlier attempt left an unfinished run directory, RENAME it (never deletes) and start fresh")
    a = ap.parse_args()
    cfg = load_config(a.config)
    from src.train_backend import SentenceTransformersBackend

    report = run_experiment(cfg, a.experiment, SentenceTransformersBackend(), smoke=a.smoke,
                            archive_incomplete=a.archive_incomplete)
    v = report["validation"]
    print(f"\n=== {report['run_id']} ({'SMOKE ' if report['smoke'] else ''}{report['status']}) ===")
    for stage, m in v["by_stage"].items():
        print(f"{stage:>13}: " + "  ".join(f"{k.replace('val_cosine_', '')}={m[k]:.4f}" for k in sorted(m) if k.endswith(("ndcg@10", "mrr@10", "recall@10"))))
    print(f"best stage: {v['best_stage']} | primary {v['primary_metric']}: {v['pre_training_primary']} -> {v['best_primary']} "
          f"(abs {v['abs_change_primary']}, rel {v['rel_change_primary_pct']}%)")
    print("gates:", {k: x for k, x in report["gates"].items() if not x} or "all True")
    print(f"train wall {report['timing_s']['train_wall']}s | peak RSS ~{report['memory_mb']['peak_rss_observed']} MB | model: {report['model_dir']}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
