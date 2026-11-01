"""Generate (and optionally execute) notebooks/09_project_summary.ipynb: the final results dashboard, rendered from the SAME generator code as the README
(src/readme_results.py), so notebook, README and reports cannot disagree. Missing stages are reported, never invented.
Usage: python scripts/build_notebook_phase09.py [--execute]"""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
CELLS = [
("md", """# Phase 9 — Project summary
Final results dashboard. Every number comes from the measured report files via `src/readme_results.py` (the same code that fills the README)."""),
("code", """import sys
from pathlib import Path
import pandas as pd, matplotlib.pyplot as plt
from IPython.display import Markdown, display
root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "configs" / "base.yaml").is_file())
sys.path.insert(0, str(root))
from src import readme_results as rr
rd = next((p for p in (root / "artifacts" / "reports", root / "results" / "reports") if (p / "test_results.json").is_file()), root / "artifacts" / "reports")
R = rr.load_reports(rd)
print("reports read from:", rd.relative_to(root), "| found:", [k for k in ("audit", "val", "test", "service", "cal", "rag") if R[k]])"""),
("md", "## Headline and results (generated)"),
("code", """display(Markdown(rr.headline(R))); display(Markdown(rr.results(R)))"""),
("md", "## Validation vs held-out test (nDCG@10 with 95% bootstrap CI)"),
("code", """if R["val"] and R["test"]:
    names = [k for k in R["test"]["models"] if k in R["val"]["models"]]
    fig, ax = plt.subplots(figsize=(6.5, 3.4)); w = 0.38
    for j, (split, getter) in enumerate((("validation", lambda k: (R["val"]["models"][k]["metrics"]["ndcg@10"], R["val"]["models"][k]["bootstrap_ci"]["ndcg@10"])),
                                         ("held-out test", lambda k: (R["test"]["models"][k]["test_metrics"]["ndcg@10"], R["test"]["models"][k]["test_bootstrap_ci"]["ndcg@10"])))):
        m = [getter(k)[0] for k in names]; lo = [getter(k)[0] - getter(k)[1]["ci_low"] for k in names]; hi = [getter(k)[1]["ci_high"] - getter(k)[0] for k in names]
        ax.bar([i + (j - 0.5) * w for i in range(len(names))], m, w, yerr=[lo, hi], capsize=3, label=split)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=10); ax.set(ylabel="nDCG@10", title="Selected model vs baseline"); ax.legend(); plt.tight_layout(); plt.show()
else:
    print("Validation and/or test results not available yet.")"""),
("md", "## Reproducibility and publication checks"),
("code", """rows = {}
for name in ("fresh_env_report_lite.json", "fresh_env_report_full.json", "repo_hygiene_report.json", "app_smoke_report.json", "search_service_report.json", "rag_validation_report.json"):
    p = rd / name
    if p.is_file():
        import json
        d = json.loads(p.read_text(encoding="utf-8")); rows[name] = {"status": d.get("status"), "mode": d.get("mode", "")}
display(pd.DataFrame(rows).T if rows else pd.DataFrame({"note": ["no verification reports found yet"]}))
print("Run: python scripts/verify_fresh_env.py | python scripts/check_repo_hygiene.py | python scripts/generate_readme_results.py --check")"""),
("md", """## Caveats
Sparse positive-only judgments; small query sets; one training seed; shared-corpus benchmark; possible overlap between the base model's pretraining mix and the corpus;
similarity is not truth; optional answers are structurally checked, not fact-checked. The dataset is CC BY-NC 2.0: see `docs/DATA_AND_LICENSES.md`."""),
]
def build() -> Path:
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s) for k, s in CELLS]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    out = ROOT / "notebooks" / "09_project_summary.ipynb"
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
