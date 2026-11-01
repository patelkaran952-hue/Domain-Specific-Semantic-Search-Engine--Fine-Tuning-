"""Scan what Git would publish for secrets, private absolute paths, oversized files and licensed dataset text.
Usage: python scripts/check_repo_hygiene.py [--mode repo|export] [--root PATH]   (exit 1 on any FAIL)
repo mode  : the working repository (licensed data is a WARNING: fine for a private repo)
export mode: strict, for the folder produced by scripts/export_portfolio.py"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import find_repo_root
from src.hygiene import scan
from src.utils import write_json


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["repo", "export"], default="repo")
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    root = Path(a.root).resolve() if a.root else find_repo_root()
    rep = scan(root, a.mode)
    if a.mode == "repo":
        write_json(rep, root / "artifacts" / "reports" / "repo_hygiene_report.json") if (root / "artifacts" / "reports").is_dir() else None
    print(f"Hygiene scan ({a.mode}) of {rep['n_files_scanned']} files: {rep['status']}")
    for k, v in sorted(rep["counts"].items()):
        print(f"  {k}: {v}")
    shown = 0
    for f in rep["findings"]:
        if shown >= 40:
            print(f"  ... {len(rep['findings']) - shown} more (see artifacts/reports/repo_hygiene_report.json)")
            break
        print(f"  [{f['severity']}] {f['kind']}  {f['path']}" + (f":{f['line']}" if f["line"] else "") + f"  {f['detail']}")
        shown += 1
    return 1 if rep["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
