"""Generate (and optionally execute) notebooks/05_model_comparison.ipynb: a thin viewer over the Phase 5 reports.
Missing stages are reported, never invented. Usage: python scripts/build_notebook_phase05.py [--execute]"""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
CELLS = [
("md", """# Phase 5 — Model comparison, validation-only selection, and the single held-out test evaluation
Everything here is read from `artifacts/reports/*`. All models are scored with the **same** implementation (`src/evaluate.py`), corpus
(5,183 docs), queries and retrieval depth. The test numbers exist only after `scripts/evaluate_models.py --stage test`."""),
("code", """import sys
from pathlib import Path
import pandas as pd, matplotlib.pyplot as plt
root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "configs" / "base.yaml").is_file())
sys.path.insert(0, str(root))
from src.config import load_config
from src.utils import read_json
cfg = load_config(["configs/baseline.yaml", "configs/compare.yaml"], root=root)
R = cfg["_abs"]["reports_dir"]
def maybe(name): p = R / name; return read_json(p) if p.is_file() else None
VAL, SEL, TEST = maybe("model_comparison_val.json"), maybe("model_selection.json"), maybe("test_results.json")
print("validation stage:", VAL["status"] if VAL else "NOT RUN (python scripts/evaluate_models.py --stage validation)")
print("selection       :", (SEL["status"] + " -> " + SEL["selected"]) if SEL else "-")
print("test stage      :", TEST["status"] if TEST else "NOT RUN (python scripts/evaluate_models.py --stage test --confirm-single-test-evaluation)")"""),
("md", "## 1. Validation comparison (162 queries; same metric code for every model)"),
("code", """if VAL:
    cols = ["ndcg@10", "recall@1", "recall@5", "recall@10", "mrr@10", "map@10", "precision@10"]
    df = pd.DataFrame({k: {c: v["metrics"][c] for c in cols} for k, v in VAL["models"].items()}).T
    df["ndcg@10 95% CI"] = [f"[{v['bootstrap_ci']['ndcg@10']['ci_low']:.4f}, {v['bootstrap_ci']['ndcg@10']['ci_high']:.4f}]" for v in VAL["models"].values()]
    display(df.round(4)); print(VAL["eval_setup"]["n_queries"], "queries,", VAL["eval_setup"]["n_judgments"], "judgments, corpus", VAL["eval_setup"]["candidate_corpus_size"], ", top_k", VAL["eval_setup"]["retrieval_top_k"])"""),
("md", "## 2. Paired differences (A minus B) with bootstrap CIs — negative changes are valid results"),
("code", """if VAL:
    rows = []
    for name, comp in VAL["paired_comparisons"].items():
        for m, c in comp.items():
            rows.append({"comparison": name, "metric": m, "mean_diff": c["mean_diff"], "ci_low": c["ci_low"], "ci_high": c["ci_high"],
                         "CI excludes 0": c["ci_excludes_zero"], "better/tied/worse": f"{c['queries_better']}/{c['queries_tied']}/{c['queries_worse']}"})
    display(pd.DataFrame(rows).round(4))"""),
("md", "## 3. Selection and freeze"),
("code", """if SEL:
    print("rule:", SEL["rule"]); print("scores:", {k: round(v, 4) for k, v in SEL["selection_scores"].items()}); print("SELECTED:", SEL["selected"], "| status:", SEL["status"])
    print("fingerprint:", SEL["fingerprint"]); print("test candidates:", SEL["test_candidates"])
    display(pd.DataFrame({k: {"weights_sha256": v["weights_sha256"][:16] + "…", "embeddings_sha256": v["embeddings_sha256"][:16] + "…", "index_sha256": v["index_sha256"][:16] + "…"} for k, v in SEL["candidates"].items()}).T)"""),
("code", """if VAL:
    fig, ax = plt.subplots(figsize=(6.5, 3.4)); names = list(VAL["models"])
    m = [VAL["models"][k]["metrics"]["ndcg@10"] for k in names]
    err = [[VAL["models"][k]["metrics"]["ndcg@10"] - VAL["models"][k]["bootstrap_ci"]["ndcg@10"]["ci_low"] for k in names],
           [VAL["models"][k]["bootstrap_ci"]["ndcg@10"]["ci_high"] - VAL["models"][k]["metrics"]["ndcg@10"] for k in names]]
    ax.bar(names, m, yerr=err, capsize=4); ax.set(ylabel="validation nDCG@10", title="Validation nDCG@10 with 95% bootstrap CI", ylim=(max(0, min(m) - 0.1), min(1, max(m) + 0.1)))
    plt.xticks(rotation=12); plt.tight_layout(); plt.show()"""),
("md", "## 4. Resource usage (measured on validation queries, CPU)"),
("code", """if VAL:
    rows = {}
    for k, v in VAL["models"].items():
        lat = v["latency_single_query"]
        rows[k] = {"encode corpus (s)": v["timing_s"]["encode_corpus"], "corpus cache reused": v["assets"]["cache_used"],
                   "query encode p50 (ms)": lat["encode_ms"]["p50"], "FAISS search p50 (ms)": lat["faiss_search_ms"]["p50"], "peak RSS (MB)": v["peak_rss_mb"]}
    display(pd.DataFrame(rows).T.round(2))
    sizes = {p.name: round(p.stat().st_size / 1e6, 2) for p in sorted((root / "artifacts" / "indexes").glob("*.faiss"))}; print("index sizes (MB):", sizes)"""),
("md", "## 5. Where do the best positives rank?"),
("code", """if VAL: display(pd.DataFrame({k: v["best_positive_rank_buckets"] for k, v in VAL["models"].items()}).T)"""),
("md", "## 6. Held-out test (frozen model + baseline, evaluated once)"),
("code", """if TEST:
    cols = ["ndcg@10", "recall@1", "recall@5", "recall@10", "mrr@10", "map@10", "precision@10"]
    df = pd.DataFrame({k: {c: v["test_metrics"][c] for c in cols} for k, v in TEST["models"].items()}).T
    df["ndcg@10 95% CI"] = [f"[{v['test_bootstrap_ci']['ndcg@10']['ci_low']:.4f}, {v['test_bootstrap_ci']['ndcg@10']['ci_high']:.4f}]" for v in TEST["models"].values()]
    display(df.round(4)); print(TEST["eval_setup"]["n_queries"], "queries,", TEST["eval_setup"]["n_judgments"], "judgments; selected:", TEST["selected_model"])
    display(pd.DataFrame({k: v["validation_to_test_change"] for k, v in TEST["models"].items()}).T.round(4))
    if TEST["paired_selected_minus_baseline"]:
        display(pd.DataFrame(TEST["paired_selected_minus_baseline"]).T[["mean_diff", "ci_low", "ci_high", "ci_excludes_zero", "queries_better", "queries_tied", "queries_worse"]].round(4))"""),
("md", "## 7. Failure cases (descriptive; the model is frozen — nothing is tuned after reading these)"),
("code", """for name in ([f"val_failure_cases_{SEL['selected']}.md"] if SEL else []) + ([f"test_failure_cases_{SEL['selected']}.md"] if TEST and SEL else []):
    p = R / name
    if p.is_file(): print("-----", name); print("\\n".join(p.read_text(encoding="utf-8").splitlines()[:30])); print()"""),
("md", """## 8. Limitations
- Judgments are sparse and positive-only; a retrieved but unjudged paper may be relevant, so scores are relative to known positives.
- SciFact is a shared-corpus benchmark (all splits search the same 5,183 papers); one training seed per model; small query sets (162 val / 300 test).
- Fine-tuned models' epoch was chosen on validation inside Phase 4, so their validation scores are optimistic; the test split is the unbiased read.
- Latency/memory are from one CPU machine and are not portable benchmarks."""),
]
def build() -> Path:
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s) for k, s in CELLS]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    out = ROOT / "notebooks" / "05_model_comparison.ipynb"
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
