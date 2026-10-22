"""Create checkpoints/phase_XX/ (README, manifest, configs, reports, snapshot, inventory) and update the root manifest.

Runs pytest + smoke test first and records the REAL outcome. Optionally commits and tags in Git.
Usage: python scripts/make_checkpoint.py --phase 1 --name "Setup, ingestion, and dataset audit" [--git]
"""
from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.utils import read_json, sha256_file, utc_now, write_json

EXCLUDE_DIRS = {".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", "checkpoints", ".venv", "venv", ".idea", ".vscode"}
EXCLUDE_FILES = {".env"}  # secrets are never inventoried or backed up
REGEN = {
    "data/raw/*": "IMMUTABLE originals — restore from your original uploads; verify SHA256 against inventory.json.",
    "data/processed/*": "python scripts/prepare_data.py",
    "artifacts/reports/data_audit.json": "python scripts/prepare_data.py",
    "artifacts/reports/sample_searches.json": "python scripts/inspect_dataset.py",
    "notebooks/01_dataset_audit.ipynb": "python scripts/build_notebook_phase01.py --execute",
}


def run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip()


def build_inventory(root: Path) -> list[dict]:
    items = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if not p.is_file() or set(rel.parts) & EXCLUDE_DIRS or p.name in EXCLUDE_FILES or p.suffix == ".pyc":
            continue
        items.append({"path": rel.as_posix(), "size_bytes": p.stat().st_size, "sha256": sha256_file(p)})
    return items


def diff_inventory(old: list[dict] | None, new: list[dict]) -> dict:
    if old is None:
        return {"added": len(new), "changed": 0, "unchanged": 0, "removed": 0}
    o, n = {i["path"]: i["sha256"] for i in old}, {i["path"]: i["sha256"] for i in new}
    return {"added": len(n.keys() - o.keys()), "removed": len(o.keys() - n.keys()),
            "changed": sum(1 for k in n.keys() & o.keys() if n[k] != o[k]),
            "unchanged": sum(1 for k in n.keys() & o.keys() if n[k] == o[k])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--git", action="store_true", help="git init/commit/tag phase-XX-complete")
    ap.add_argument("--force", action="store_true", help="overwrite an existing checkpoint of this phase")
    a = ap.parse_args()

    cfg = load_config()
    root = cfg["_root"]
    ck_root = cfg["_abs"]["checkpoints_dir"]
    ck = ck_root / f"phase_{a.phase:02d}"
    if ck.exists() and not a.force:
        print(f"{ck} already exists. Re-run with --force to replace this checkpoint.")
        return 2

    # quality gates (real execution) ------------------------------------------------------
    rc_t, out_t = run([sys.executable, "-m", "pytest", "-q"], root)
    rc_s, out_s = run([sys.executable, "scripts/smoke_test.py"], root)
    audit = read_json(cfg["_abs"]["reports_dir"] / "data_audit.json")
    gates = {"pytest": "PASS" if rc_t == 0 else "FAIL", "smoke_test": "PASS" if rc_s == 0 else "FAIL",
             "data_audit": audit["status"]}
    status = "PASS" if all(v == "PASS" for v in gates.values()) else "FAIL"

    # copy small indispensable files ------------------------------------------------------
    if ck.exists():
        shutil.rmtree(ck)
    for sub in ("configs", "reports", "snapshot"):
        (ck / sub).mkdir(parents=True)
    for f in (root / "configs").glob("*.yaml"):
        shutil.copyfile(f, ck / "configs" / f.name)
    for f in cfg["_abs"]["reports_dir"].glob("*"):
        if f.is_file():
            shutil.copyfile(f, ck / "reports" / f.name)
    (ck / "reports" / "pytest_output.txt").write_text(out_t + "\n", encoding="utf-8")
    (ck / "reports" / "smoke_test_output.txt").write_text(out_s + "\n", encoding="utf-8")
    for n in ("train_query_ids.json", "val_query_ids.json", "test_query_ids.json", "document_ids.json",
              "train_qrels.tsv", "val_qrels.tsv"):
        shutil.copyfile(cfg["_abs"]["processed_dir"] / n, ck / "snapshot" / n)

    # inventory (complete project, not just this phase) ------------------------------------
    inv = build_inventory(root)
    prev_path = ck_root / "manifest.json"
    prev_inv = None
    if prev_path.is_file():
        prev = read_json(prev_path)
        pi = ck_root / f"phase_{prev['latest_phase']:02d}" / "inventory.json"
        if pi.is_file() and prev["latest_phase"] != a.phase:
            prev_inv = read_json(pi)["files"]
    change = diff_inventory(prev_inv, inv)
    write_json({"generated_utc": utc_now(), "phase": a.phase, "n_files": len(inv), "files": inv}, ck / "inventory.json")

    import numpy, pandas, sklearn, yaml  # noqa
    manifest = {
        "phase": a.phase, "name": a.name, "status": status, "timestamp_utc": utc_now(),
        "quality_gates": gates, "changes_since_previous_checkpoint": change,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "numpy": numpy.__version__, "pandas": pandas.__version__,
                        "scikit_learn": sklearn.__version__, "pyyaml": yaml.__version__},
        "seed": cfg["project"]["seed"],
        "key_hashes": {r["path"]: r["sha256"] for r in inv if r["path"].startswith(("data/raw/", "data/processed/"))},
        "regeneration": REGEN,
        "git_tag": f"phase-{a.phase:02d}-complete" if a.git else None,
        "not_in_git_note": "data/raw is git-ignored: restore originals and verify hashes. No models/embeddings exist yet.",
    }
    write_json(manifest, ck / "manifest.json")
    write_json({"latest_phase": a.phase, "latest_status": status, "timestamp_utc": manifest["timestamp_utc"],
                "checkpoint_dir": f"checkpoints/phase_{a.phase:02d}"}, prev_path)

    readme = f"""# Phase {a.phase:02d} checkpoint — {a.name}

**Status: {status}** | generated {manifest['timestamp_utc']}

## Quality gates (real executions)
| Gate | Result |
|---|---|
| pytest | {gates['pytest']} |
| smoke_test.py | {gates['smoke_test']} |
| data audit | {gates['data_audit']} |

## File changes vs previous checkpoint
{change}

## Resume / reproduce (Windows PowerShell, repo root)
```powershell
conda activate scifact-search
python scripts/prepare_data.py
python scripts/inspect_dataset.py
python scripts/smoke_test.py
python -m pytest -q
python scripts/build_notebook_phase01.py --execute
```
Verify integrity: compare SHA256 values in `inventory.json` (`Get-FileHash <file> -Algorithm SHA256`).

## Contents
`manifest.json` (versions, hashes, regeneration), `inventory.json` (every project file: path/size/SHA256),
`configs/`, `reports/` (audit, sample searches, test outputs, environment freeze), `snapshot/` (split IDs and qrels).

## Next phase
Phase {a.phase + 1} — requires explicit user approval ("Proceed to Phase {a.phase + 1}").
"""
    (ck / "README.md").write_text(readme, encoding="utf-8")

    if a.git:
        if not (root / ".git").exists():
            run(["git", "init", "-b", "main"], root)
        ident = ["-c", "user.name=scifact-project", "-c", "user.email=noreply@example.invalid"]
        run(["git", "add", "-A"], root)
        rc, out = run(["git", *ident, "commit", "-m", f"Phase {a.phase:02d} complete: {a.name}"], root)
        run(["git", *ident, "tag", "-f", f"phase-{a.phase:02d}-complete"], root)
        print("git:", out.splitlines()[0] if out else rc)

    print(f"Checkpoint {ck} written | status={status} | gates={gates} | files inventoried={len(inv)} | change={change}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
