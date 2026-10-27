"""Phase 7 smoke test: run the REAL app script headlessly against the REAL packaged model + FAISS index and simulate user actions.
Optionally also start `streamlit run` and check the server health endpoint.
Usage: python scripts/smoke_test_app.py [--serve-check] [--force]      (needs streamlit; loads the model, ~20-40 s)"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import ui_support as ui
from src.app_smoke import run_app_smoke, serve_check
from src.utils import write_json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve-check", action="store_true", help="also launch `streamlit run` headlessly and poll its health endpoint")
    ap.add_argument("--force", action="store_true", help="replace an existing app_smoke_report.json")
    a = ap.parse_args()
    cfg = ui.get_config()
    root, report_p = cfg["_root"], cfg["_abs"]["reports_dir"] / "app_smoke_report.json"
    if report_p.exists() and not a.force:
        raise SystemExit(f"{report_p.name} already exists. Re-run with --force to replace it.")
    opts = ui.discover_models(cfg)
    if not opts or not opts[0].available:
        print(f"BLOCKED: {opts[0].reason if opts else 'no frozen selection'}")
        return 4
    print(f"Models offered: {[(o.key, o.available) for o in opts]}")
    rep = run_app_smoke(cfg, root / "app" / "streamlit_app.py", cfg["smoke"]["apptest_timeout_s"], cfg["smoke"]["queries"], int(cfg["service"]["default_top_k"]))
    rep["models_offered"] = [{"key": o.key, "available": o.available} for o in opts]
    if a.serve_check:
        rep["steps"].append(serve_check(root, "app/streamlit_app.py", cfg["app"]["server"]["port"], cfg["app"]["server"]["health_timeout_s"]))
        rep["status"] = "PASS" if all(s["ok"] for s in rep["steps"]) else "FAIL"
    rep["manual_check_pending"] = "Open `streamlit run app/streamlit_app.py` in a browser and visually confirm layout (cannot be automated here)."
    write_json(rep, report_p)
    print(f"\n=== App smoke test: {rep['status']} ===")
    for s in rep["steps"]:
        print(f"  {'PASS' if s['ok'] else 'FAIL'}  {s['step']:<44} {s['seconds']:>6.1f}s  {s['detail'][:110]}")
    return 0 if rep["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
