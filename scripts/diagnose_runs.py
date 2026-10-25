"""Read-only: show how far each fine-tuning attempt got (files, logs, recorded failure). Writes nothing.
Usage: python scripts/diagnose_runs.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.diagnose import describe_environment, describe_pretrained_cache, describe_run


def main() -> int:
    cfg = load_config(["configs/baseline.yaml", "configs/finetune.yaml"])
    models, reports, root = cfg["_abs"]["models_dir"], cfg["_abs"]["reports_dir"], cfg["_root"]
    lines = describe_environment() + describe_pretrained_cache(reports / "finetune_pretrained_val_evaluator.json")
    for e in cfg["experiments"].values():
        rid = e["run_id"]
        lines += describe_run(models / rid, reports / f"finetune_{rid}_report.json")
        lines += describe_run(root / "artifacts" / "smoke" / "finetune" / rid, root / "artifacts" / "smoke" / f"finetune_{rid}_report.json") \
            if (root / "artifacts" / "smoke" / "finetune" / rid).exists() else []
        for extra in sorted(models.glob(f"{rid}.incomplete-*")):
            lines += describe_run(extra, reports / "_none_")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
