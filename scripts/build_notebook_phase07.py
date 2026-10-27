"""Generate (and optionally execute) notebooks/07_streamlit_app.ipynb: viewer over artifacts/reports/app_smoke_report.json plus how to launch.
Missing stages are reported, never invented. Usage: python scripts/build_notebook_phase07.py [--execute]"""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
CELLS = [
("md", """# Phase 7 — Streamlit UI
Viewer over `artifacts/reports/app_smoke_report.json`. The app itself is `app/streamlit_app.py`; launch it with
`streamlit run app/streamlit_app.py`. Nothing is computed here."""),
("code", """import sys
from pathlib import Path
import pandas as pd
root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "configs" / "base.yaml").is_file())
sys.path.insert(0, str(root))
from src import ui_support as ui
from src.utils import read_json
cfg = ui.get_config()
rp = cfg["_abs"]["reports_dir"] / "app_smoke_report.json"
REP = read_json(rp) if rp.is_file() else None
print("app smoke test:", REP["status"] if REP else "NOT RUN (python scripts/smoke_test_app.py --serve-check)")
opts = ui.discover_models(cfg)
display(pd.DataFrame([{"model": o.key, "label": o.label, "available": o.available, "reason": o.reason or "-"} for o in opts]) if opts else pd.DataFrame({"note": ["no frozen selection found"]}))"""),
("md", "## 1. What the app does"),
("md", """- **Inputs:** a text query and a top-k slider (1–50). Example queries fill the box in one click.
- **Each result:** rank, title, **paper ID** (SciFact corpus ID, for citation), snippet, **cosine similarity**, and an expandable full abstract.
- **Graceful states:** empty query → prompt; clipped/long queries → warnings; load failures → readable error, never a traceback.
- **Models:** the frozen fine-tuned model by default. The pretrained baseline (and a side-by-side comparison) appears only if its index/embeddings exist.
- **Performance:** each model loads once per process (`st.cache_resource`), with one warm-up query; the baseline loads only when selected."""),
("md", "## 2. Headless smoke test (real app script, real package; not a browser)"),
("code", """if REP:
    display(pd.DataFrame(REP["steps"])[["step", "ok", "seconds", "detail"]])
    print("method:", REP["method"]); print("models offered:", REP.get("models_offered")); print("manual check pending:", REP.get("manual_check_pending"))"""),
("md", """## 3. Launch (PowerShell, repo root)
```powershell
streamlit run app/streamlit_app.py
```
The first page load takes roughly 20 s while the model and index load; later searches take tens of milliseconds (see Phase 6 benchmark)."""),
("md", """## 4. Caveats
- The smoke test drives the app script headlessly; visual layout still needs a human look in a browser.
- Scores are cosine similarities, not probabilities; results reflect similarity to 5,183 SciFact abstracts only.
- Research prototype; not medical or clinical advice."""),
]
def build() -> Path:
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s) for k, s in CELLS]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    out = ROOT / "notebooks" / "07_streamlit_app.ipynb"
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
