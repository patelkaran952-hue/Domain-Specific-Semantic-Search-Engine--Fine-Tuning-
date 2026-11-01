"""Fresh-environment verification: create a NEW virtual environment, install the project's requirements, run the test suite inside it.
--lite (default): skips torch / sentence-transformers / transformers / datasets / accelerate / jupyter (the whole suite is designed to run without them)
--full          : installs everything (large download) and also runs the data smoke test if processed data exists
Usage: python scripts/verify_fresh_env.py [--lite|--full] [--keep]"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import find_repo_root
from src.envtools import filter_lite, parse_requirements, venv_python
from src.utils import utc_now, write_json


def run(cmd, cwd=None, timeout=3600):
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr), time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--lite", action="store_true")
    g.add_argument("--full", action="store_true")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    mode = "full" if a.full else "lite"
    root = find_repo_root()
    reqs = [x for f in ("requirements.txt", "requirements-dev.txt") if (root / f).is_file() for x in parse_requirements(root / f)]
    lines = [orig for _, orig in reqs] if mode == "full" else filter_lite(reqs)
    steps, tmp, venv_ok = [], Path(tempfile.mkdtemp(prefix="fresh-env-")), False
    venv = tmp / "venv"
    try:
        rc, out, s = run([sys.executable, "-m", "venv", str(venv)]); steps.append({"step": "create venv", "ok": rc == 0, "seconds": round(s, 1)})
        venv_ok = rc == 0
        py = venv_python(venv)
        if rc == 0:
            (tmp / "req.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            rc, out, s = run([str(py), "-m", "pip", "install", "-q", "--disable-pip-version-check", "-r", str(tmp / "req.txt")])
            steps.append({"step": f"pip install ({len(lines)} requirement lines, {mode})", "ok": rc == 0, "seconds": round(s, 1), "tail": out[-600:] if rc else ""})
        if rc == 0:
            rc, out, s = run([str(py), "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=root)
            steps.append({"step": "pytest in the fresh environment", "ok": rc == 0, "seconds": round(s, 1), "summary": (out.strip().splitlines() or [""])[-1]})
            if rc == 0 and mode == "full" and (root / "data" / "processed" / "document_ids.json").is_file():
                rc, out, s = run([str(py), "scripts/smoke_test.py"], cwd=root)
                steps.append({"step": "scripts/smoke_test.py", "ok": rc == 0, "seconds": round(s, 1)})
        pl = run([str(py), "-m", "pip", "list", "--format=json"])[1] if venv_ok else "[]"
        try:
            pkgs = {p["name"]: p["version"] for p in json.loads(pl[pl.index("["):])}
        except Exception:  # noqa: BLE001
            pkgs = {}
    finally:
        if not a.keep:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
    rep = {"status": "PASS" if steps and all(s["ok"] for s in steps) else "FAIL", "mode": mode, "generated_utc": utc_now(), "python": sys.version.split()[0],
           "steps": steps, "installed_in_fresh_env": pkgs,
           "note": "lite mode skips the heavy ML stack: it proves the code and tests do not rely on undeclared packages; full mode proves a complete install."}
    (root / "artifacts" / "reports").mkdir(parents=True, exist_ok=True)
    write_json(rep, root / "artifacts" / "reports" / f"fresh_env_report_{mode}.json")
    print(f"Fresh-environment verification ({mode}): {rep['status']}")
    for s in steps:
        print(f"  {'PASS' if s['ok'] else 'FAIL'}  {s['step']:<56} {s['seconds']:>7.1f}s  {s.get('summary', '')}")
        if s.get("tail"):
            print("   ", s["tail"].replace("\n", "\n    "))
    return 0 if rep["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
