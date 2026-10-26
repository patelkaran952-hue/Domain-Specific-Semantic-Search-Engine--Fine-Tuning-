"""Generate (and optionally execute) notebooks/06_search_service.ipynb: a thin viewer over artifacts/reports/search_service_report.json
and the package manifest. Missing stages are reported, never invented. Usage: python scripts/build_notebook_phase06.py [--execute]"""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
CELLS = [
("md", """# Phase 6 — Packaged search service
Viewer over `service_manifest.json` and `artifacts/reports/search_service_report.json`. No search is run here; use
`python scripts/search_cli.py "query"` or `SearchService.load(...)`."""),
("code", """import sys
from pathlib import Path
import pandas as pd
root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "configs" / "base.yaml").is_file())
sys.path.insert(0, str(root))
from src.config import load_config
from src.utils import read_json
cfg = load_config(["configs/baseline.yaml", "configs/compare.yaml", "configs/service.yaml"], root=root)
pkg = cfg["_abs"]["service_dir"] / cfg["service"]["package_name"]
mp, rp = pkg / "service_manifest.json", cfg["_abs"]["reports_dir"] / "search_service_report.json"
MAN = read_json(mp) if mp.is_file() else None
REP = read_json(rp) if rp.is_file() else None
print("package :", "built" if MAN else "NOT BUILT (python scripts/build_search_package.py)")
print("validation:", REP["status"] if REP else "NOT RUN (python scripts/validate_search_service.py)")"""),
("md", "## 1. Package contents and provenance"),
("code", """if MAN:
    display(pd.Series({k: MAN[k] for k in ("package_name", "selected_model", "label", "n_documents", "embedding_dim", "max_seq_length", "normalization", "metric", "index_type", "created_utc")}, name="value").to_frame())
    print("freeze fingerprint:", MAN["freeze_fingerprint"]); print("weights sha256   :", MAN["weights_sha256"])
    print("reported metrics (from Phase 5, not recomputed here):")
    display(pd.DataFrame({k: v for k, v in MAN["reported_metrics"].items() if k != "source"}).loc[["ndcg@10", "recall@1", "recall@10", "mrr@10"]].round(4))
    print(MAN["notice"])"""),
("md", "## 2. Exit gates (all must be True)"),
("code", """if REP: display(pd.Series(REP["gates"], name="passed").to_frame())"""),
("md", "## 3. Reload consistency vs Phase 5 rankings (tie-aware)"),
("code", """if REP:
    print(REP["reload_consistency"]); print(REP["end_to_end_metric"]); print("max |service score - independent dot product|:", REP["score_check_max_abs_diff"])"""),
("md", "## 4. Latency AFTER startup (CPU, sequential, one user) and startup cost"),
("code", """if REP:
    rows = {k: {**{f"total {s} (ms)": v for s, v in b["total_ms"].items()}, "encode mean (ms)": b["encode_ms_mean"], "FAISS mean (ms)": b["faiss_search_ms_mean"], "format mean (ms)": b["format_ms_mean"], "queries/s": b["throughput_qps_sequential"]} for k, b in REP["latency_after_startup"].items()}
    display(pd.DataFrame(rows).T.round(2))
    print("startup seconds (EXCLUDED from the latency above):", {k: round(v, 2) for k, v in REP["startup_s_excluded_from_latency"].items()})
    print("memory:", REP["memory_mb"]); print("package:", REP["package"]); print("environment:", REP["environment"])"""),
("md", "## 5. Observed input handling"),
("code", """if REP: display(pd.Series(REP["input_handling_observed"], name="behaviour").to_frame())"""),
("md", "## 6. Example queries (retrieval demo — similarity is not a probability that a claim is true)"),
("code", """if REP:
    for q, hits in REP["example_queries"].items():
        print(q)
        for h in hits: print(f"  #{h['rank']} [{h['corpus_id']}] cos={h['similarity_score']:.3f}  {h['title'][:90]}")"""),
("md", """## 7. Caveats
- Latency is for one CPU machine, sequential single-user calls; it is not a throughput or multi-user guarantee.
- The service searches only the 5,183 SciFact abstracts it was built on; it has no knowledge beyond them.
- Results reflect semantic similarity to known abstracts, not verification of any claim."""),
]
def build() -> Path:
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s) for k, s in CELLS]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    out = ROOT / "notebooks" / "06_search_service.ipynb"
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
