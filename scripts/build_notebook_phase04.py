"""Generate (and optionally execute) notebooks/04_finetuning.ipynb: a thin viewer over the artifacts written by
scripts/finetune.py and scripts/verify_finetune.py. Missing runs are reported, never invented.
Usage: python scripts/build_notebook_phase04.py [--execute]"""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
CELLS = [
("md", """# Phase 4 — MiniLM fine-tuning (validation only)
Viewer over `artifacts/reports/finetune_*_report.json` and each run's `train_log_history.json`.
Validation numbers come from the Sentence-Transformers `InformationRetrievalEvaluator` (same evaluator for epoch 0 and every epoch);
Phase 5 re-scores every model with the project's own metric code. `test.tsv` is not used."""),
("code", """import sys
from pathlib import Path
import pandas as pd, matplotlib.pyplot as plt
root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "configs" / "base.yaml").is_file())
sys.path.insert(0, str(root))
from src.config import load_config
from src.utils import read_json
cfg = load_config(["configs/baseline.yaml", "configs/finetune.yaml"], root=root)
reports = cfg["_abs"]["reports_dir"]
REP = {}
for key, e in cfg["experiments"].items():
    p = reports / f"finetune_{e['run_id']}_report.json"
    if p.is_file(): REP[key] = read_json(p)
    print(f"{key:>14} ({e['run_id']}):", REP[key]["status"] if key in REP else "NOT RUN YET (python scripts/finetune.py --experiment %s)" % key)
vp = reports / "finetune_verification.json"
print("verification:", read_json(vp)["status"] if vp.is_file() else "not run")"""),
("md", "## 1. Configuration shared by both runs"),
("code", """if REP: display(pd.Series(next(iter(REP.values()))["hyperparameters"], name="value").to_frame())"""),
("md", "## 2. Validation metrics by stage (evaluator-defined; epoch 0 = pretrained, same evaluator)"),
("code", """rows = []
for k, r in REP.items():
    for stage, m in r["validation"]["by_stage"].items():
        rows.append({"run": r["run_id"], "stage": stage, **{n.replace("val_cosine_", ""): v for n, v in m.items()}})
if rows:
    df = pd.DataFrame(rows); display(df.round(4))"""),
("code", """if REP:
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for k, r in REP.items():
        pm = r["validation"]["primary_metric"]
        stages = list(r["validation"]["by_stage"])
        ax.plot(range(len(stages)), [r["validation"]["by_stage"][s][pm] for s in stages], marker="o", label=r["run_id"])
        ax.set_xticks(range(len(stages))); ax.set_xticklabels(stages, rotation=20)
    ax.set(ylabel="validation nDCG@10 (evaluator)", title="Pretrained vs fine-tuned, by epoch"); ax.legend(); plt.tight_layout(); plt.show()"""),
("md", "## 3. Change vs pretrained (negative values are valid results)"),
("code", """if REP:
    display(pd.DataFrame({r["run_id"]: {"best_stage": r["validation"]["best_stage"], "pretrained": r["validation"]["pre_training_primary"],
        "best": r["validation"]["best_primary"], "abs_change": r["validation"]["abs_change_primary"],
        "rel_change_%": r["validation"]["rel_change_primary_pct"]} for r in REP.values()}))"""),
("md", "## 4. Training loss, runtime and memory"),
("code", """import json
if REP:
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for k, r in REP.items():
        hist = read_json(root / "artifacts" / "models" / r["run_id"] / "train_log_history.json")
        pts = [(h["step"], h["loss"]) for h in hist if "loss" in h and "step" in h]
        if pts: ax.plot(*zip(*pts), label=r["run_id"])
    ax.set(xlabel="step", ylabel="train loss"); ax.legend(); plt.tight_layout(); plt.show()
    display(pd.DataFrame({r["run_id"]: {**r["timing_s"], **r["memory_mb"]} for r in REP.values()}))"""),
("md", "## 5. Exit gates"),
("code", """if REP: display(pd.DataFrame({r["run_id"]: r["gates"] for r in REP.values()}))"""),
("md", """## 6. Caveats
- Selecting the best epoch inside a run uses validation; the final model between runs is chosen in Phase 5 on validation, then tested once.
- Mined negatives are unjudged; a worse hard-negative run is a legitimate outcome (false negatives).
- CPU training is only approximately reproducible across library versions."""),
]
def build() -> Path:
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s) for k, s in CELLS]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    out = ROOT / "notebooks" / "04_finetuning.ipynb"
    nbf.write(nb, out)
    return out
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--execute", action="store_true"); a = ap.parse_args()
    path = build()
    if a.execute:
        from nbconvert.preprocessors import ExecutePreprocessor
        nb = nbf.read(path, as_version=4)
        ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(ROOT)}})
        nbf.write(nb, path)
    print("wrote", path)
