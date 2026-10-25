"""Generate (and optionally execute) notebooks/03_negative_mining.ipynb: a thin viewer over the artifacts of
scripts/mine_negatives.py and scripts/verify_mining.py. Missing sources are reported, never invented.
Usage: python scripts/build_notebook_phase03.py [--execute]"""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
CELLS = [
("md", """# Phase 3 — Positive pairs and hard-negative mining (training partition only)
Viewer over `data/mining/*` and `artifacts/reports/negative_mining_*`. **Mined negatives are unjudged top-ranked
documents, not verified irrelevant ones.** Only training qrels were read; validation/test judgments were not."""),
("code", """import sys, json
from pathlib import Path
import pandas as pd, matplotlib.pyplot as plt
root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "configs" / "base.yaml").is_file())
sys.path.insert(0, str(root))
from src.config import load_config
from src.negative_mining import read_jsonl_rows
from src.utils import read_json
cfg = load_config(["configs/baseline.yaml", "configs/mining.yaml"], root=root)
mdir, reports = cfg["_abs"]["mining_dir"], cfg["_abs"]["reports_dir"]
SOURCES = cfg["mining"]["sources"]
REP = {s: read_json(reports / f"negative_mining_{s}_report.json") for s in SOURCES if (reports / f"negative_mining_{s}_report.json").is_file()}
for s in SOURCES:
    print(f"{s:>6}:", REP[s]["status"] if s in REP else "NOT RUN YET (run: python scripts/mine_negatives.py --source %s)" % s)
vp = reports / "negative_mining_verification.json"
print("verification:", read_json(vp)["status"] if vp.is_file() else "not run")"""),
("md", "## 1. Positive pairs (training partition)"),
("code", """pairs = read_jsonl_rows(mdir / "train_positive_pairs.jsonl")
dfp = pd.DataFrame(pairs)
print(len(pairs), "pairs from", dfp.query_id.nunique(), "training queries; positives/query:", dfp.groupby("query_id").size().value_counts().sort_index().to_dict())"""),
("md", "## 2. Exit gates per source (all must be True)"),
("code", """if REP: display(pd.DataFrame({s: r["gates"] for s, r in REP.items()}))"""),
("md", "## 3. Score distributions and false-negative risk"),
("code", """rows = {}
for s, r in REP.items():
    st = r["stats"]
    rows[s] = {"pairs": r["n_positive_pairs"], "triplets": r["n_triplets"], "unique_neg_docs": st["unique_negative_docs"],
               "neg score median": st["negative_score"]["median"], "best-pos score median": st["best_positive_score"]["median"],
               "neg/pos ratio median": st["negative_to_positive_score_ratio"]["median"],
               "flagged FN-risk frac (ratio>=%.2f)" % st["flagged_false_negative_risk"]["threshold"]: st["flagged_false_negative_risk"]["fraction"],
               "neg that are positives of other train queries": st["negatives_that_are_positives_of_OTHER_train_queries"]}
if rows: display(pd.DataFrame(rows).T)"""),
("code", """fig, axes = plt.subplots(1, max(1, len(REP)), figsize=(5.2 * max(1, len(REP)), 3.2), squeeze=False)
for ax, s in zip(axes[0], REP):
    ratios = [n["score_ratio"] for m in read_jsonl_rows(mdir / f"hard_negatives_{s}.jsonl") for n in m["negatives"] if n["score_ratio"] is not None]
    ax.hist([min(r, 3) for r in ratios], bins=40); ax.axvline(1.0, color="r", ls="--", label="= best positive")
    ax.set(title=f"{s}: negative / best-positive score (clipped at 3)", xlabel="ratio", ylabel="pooled negatives"); ax.legend()
plt.tight_layout(); plt.show()"""),
("md", "## 4. Retriever diagnostics on TRAINING queries (not validation metrics)"),
("code", """if REP: display(pd.DataFrame({s: r["stats"]["train_query_best_positive_rank_diagnostics"] for s, r in REP.items()}))"""),
("md", "## 5. Manual inspection samples (read for false negatives)"),
("code", """for s in REP:
    txt = (mdir / f"samples_{s}.md").read_text(encoding="utf-8")
    print(f"----- {s} (first ~3 queries) -----"); print("\\n".join(txt.splitlines()[:34])); print()"""),
("md", """## 6. Caveats
- Unjudged ≠ irrelevant: a negative that out-scores the labeled positive may be genuinely relevant (SciFact qrels are sparse).
- Negatives that are positives of *other* training queries are legitimate for this query but may be topically close.
- Training-query rank diagnostics are in-sample for the retriever's index and are not performance claims."""),
]
def build() -> Path:
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s) for k, s in CELLS]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    out = ROOT / "notebooks" / "03_negative_mining.ipynb"
    nbf.write(nb, out)
    return out
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--execute", action="store_true"); a = ap.parse_args()
    path = build()
    if a.execute:
        from nbconvert.preprocessors import ExecutePreprocessor
        nb = nbf.read(path, as_version=4)
        ExecutePreprocessor(timeout=300, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(ROOT)}})
        nbf.write(nb, path)
    print("wrote", path)
