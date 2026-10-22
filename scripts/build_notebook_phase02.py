"""Generate (and optionally execute) notebooks/02_baseline_retrieval.ipynb. Thin viewer over
artifacts/reports/baseline_val_*.{json,csv} written by scripts/run_baseline.py (no notebook-only logic).
Usage: python scripts/build_notebook_phase02.py [--execute]"""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
CELLS = [
("md", """# Phase 2 — Baseline MiniLM Retrieval (validation only)
Viewer over the outputs of `python scripts/run_baseline.py`. **Nothing is computed or invented here**:
if the report is missing, the notebook says so. `test.tsv` is never used in this phase."""),
("code", """import sys
from pathlib import Path
import pandas as pd, matplotlib.pyplot as plt
root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "configs" / "base.yaml").is_file())
sys.path.insert(0, str(root))
from src.config import load_config
from src.data_io import read_jsonl
from src.utils import read_json
cfg = load_config("configs/baseline.yaml", root=root)
reports, proc = cfg["_abs"]["reports_dir"], cfg["_abs"]["processed_dir"]
rp = reports / "baseline_val_metrics.json"
HAVE = rp.is_file()
print("baseline report found:" , HAVE)
if not HAVE:
    print("Run `python scripts/run_baseline.py` first, then re-run this notebook.")
else:
    rep = read_json(rp); print("status:", rep["status"], "| model:", rep["model"]["model_name"], "@", rep["model"]["resolved_revision"])"""),
("md", "## 1. Validation metrics (baseline)"),
("code", """if HAVE:
    display(pd.Series(rep["metrics"], name="baseline (val)").to_frame().round(4))
    display(pd.Series(rep["eval_setup"]).to_frame("value"))"""),
("md", "## 2. Integrity checks (cosine = inner product, index/ID alignment, reproducibility)"),
("code", """if HAVE: display(pd.Series(rep["checks"], name="value").to_frame())"""),
("md", "## 3. Truncation (exact, from the MiniLM tokenizer), latency and memory"),
("code", """if HAVE:
    print("truncation:", rep["truncation"])
    print("timing_s:", rep["timing_s"])
    print("single-query latency (ms):", rep["latency_single_query"])
    print("memory:", rep["memory_mb"])"""),
("md", "## 4. Per-query nDCG@10 distribution and failure cases"),
("code", """if HAVE:
    pq = pd.read_csv(reports / "baseline_val_per_query.csv", dtype={"query_id": str})
    plt.figure(figsize=(6, 3)); plt.hist(pq["ndcg@10"], bins=20); plt.title("Per-query nDCG@10 (validation)"); plt.xlabel("nDCG@10"); plt.ylabel("queries"); plt.show()
    print("queries with no positive in top 10:", int((pq["recall@10"] == 0).sum()), "of", len(pq))
    docs = {d["_id"]: d["title"] for d in read_jsonl(proc / "corpus_processed.jsonl")}
    qtext = {r["_id"]: r["text"] for r in read_jsonl(proc / "queries_processed.jsonl")}
    run = read_json(reports / "baseline_val_run_top100.json")
    from src.data_io import read_qrels
    pos = {}
    for q, d, _ in read_qrels(proc / "val_qrels.tsv"): pos.setdefault(q, set()).add(d)
    for q in pq.sort_values("ndcg@10").head(5)["query_id"]:
        print(f"\\nQuery {q}: {qtext[q]}")
        print("  judged positive:", [docs[d][:70] for d in pos[q]])
        print("  top-3 retrieved:", [(d, round(s, 3), docs[d][:55]) for d, s in run[q][:3]])
        print("  (retrieved docs may be relevant but unjudged: qrels are sparse)")"""),
("md", """## 5. Caveats
- Metrics are relative to sparse, positive-only judgments; an unjudged retrieved paper is not necessarily irrelevant.
- Similarity scores are cosine similarities, not probabilities of factual correctness.
- Latency/memory were measured on the machine that ran the script; they are not portable benchmarks."""),
]
def build() -> Path:
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s) for k, s in CELLS]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    out = ROOT / "notebooks" / "02_baseline_retrieval.ipynb"
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
