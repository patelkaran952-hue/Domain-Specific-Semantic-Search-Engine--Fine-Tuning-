"""Generate (and optionally execute) notebooks/01_dataset_audit.ipynb.
The notebook is a thin viewer: all logic lives in src/ and scripts/ (no notebook-only functionality).
Usage: python scripts/build_notebook_phase01.py [--execute]"""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

CELLS = [
("md", """# Phase 1 — Dataset Audit (SciFact)
Thin viewer over artifacts produced by `scripts/prepare_data.py` and `scripts/inspect_dataset.py`.
No model is trained here. Run those scripts first."""),
("code", """import sys, json
from pathlib import Path
import pandas as pd, matplotlib.pyplot as plt
root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "configs" / "base.yaml").is_file())
sys.path.insert(0, str(root))
from src.config import load_config
from src.data_io import read_jsonl, read_qrels
from src.utils import read_json
cfg = load_config(root=root)
proc, reports = cfg["_abs"]["processed_dir"], cfg["_abs"]["reports_dir"]
audit = read_json(reports / "data_audit.json")
print("Audit status:", audit["status"], "| seed:", audit["seed"])"""),
("md", "## 1. Raw file integrity (SHA-256)"),
("code", """pd.Series(audit["raw_file_sha256"], name="sha256").to_frame()"""),
("md", "## 2. Counts and reconciliation"),
("code", """pd.Series(audit["reconciliation"], name="value").to_frame()"""),
("md", "## 3. Leakage gates (all must be True)"),
("code", """pd.Series(audit["leakage_gates"], name="passed").to_frame()"""),
("md", """## 4. Text length statistics (whitespace words)
Encoder truncation (256 wordpieces) will be measured with the real tokenizer in Phase 2; here is only a heuristic."""),
("code", """rows = {"abstract_words": audit["corpus"]["abstract_words"], "title_words": audit["corpus"]["title_words"],
        "query_words": audit["queries"]["query_words"]}
display(pd.DataFrame(rows).T)
print(audit["truncation_heuristic"])"""),
("code", """docs = read_jsonl(proc / "corpus_processed.jsonl")
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
ax[0].hist([len(d["doc_text"].split()) for d in docs], bins=50)
ax[0].axvline(cfg["preprocess"]["approx_words_per_wordpiece_budget"], color="r", ls="--", label="heuristic 256-wordpiece budget")
ax[0].set(title="Document length (words)", xlabel="words", ylabel="documents"); ax[0].legend()
h = audit["train_qrels_official"]["positives_per_query_histogram"]
ax[1].bar(list(h.keys()), list(h.values())); ax[1].set(title="Positives per training query", xlabel="# positives", ylabel="queries")
plt.tight_layout(); plt.show()"""),
("md", "## 5. Official qrels vs. derived split"),
("code", """summary = pd.DataFrame({
    "queries": [audit["reconciliation"]["split_train_queries"], audit["reconciliation"]["split_val_queries"], audit["reconciliation"]["official_test_queries"]],
    "positive_rows": [audit["reconciliation"]["split_train_rows"], audit["reconciliation"]["split_val_rows"], audit["test_qrels_official"]["n_rows"]],
}, index=["train (80%)", "validation (20%)", "test (reserved)"])
summary"""),
("md", """## 6. Query metadata is NOT a relevance label
`SUPPORT` / `CONTRADICT` annotations describe claim verification, not retrieval relevance (info only)."""),
("code", """audit["queries"]["metadata_annotation_label_counts_INFO_ONLY"]"""),
("md", """## 7. Sample lexical searches (TF-IDF, training queries only)
A training-free sanity probe — **not** the project baseline and not a reported metric. `*` marks a judged positive."""),
("code", """samples = read_json(reports / "sample_searches.json")["samples"]
for s in samples:
    print(f"Query {s['query_id']}: {s['query']}")
    for h in s["top_k"]:
        print(f"  {'*' if h['corpus_id'] in s['positives'] else ' '} #{h['rank']} [{h['corpus_id']}] {h['score']:.3f} {h['title'][:75]}")
    print()"""),
("md", """## 8. Caveats
- Qrels are positive-only and sparse; unjudged documents are *unlabeled*, not confirmed negatives.
- The corpus is shared across train/val/test (shared-corpus benchmark).
- `test.tsv` is reserved for one final evaluation in Phase 5."""),
]


def build() -> Path:
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s) for k, s in CELLS]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    out = ROOT / "notebooks" / "01_dataset_audit.ipynb"
    out.parent.mkdir(exist_ok=True)
    nbf.write(nb, out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    path = build()
    if a.execute:
        from nbconvert.preprocessors import ExecutePreprocessor

        nb = nbf.read(path, as_version=4)
        ExecutePreprocessor(timeout=300, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(ROOT)}})
        nbf.write(nb, path)
    print("wrote", path)
