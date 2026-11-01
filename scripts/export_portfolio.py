"""Build a publishable copy of the project in dist/portfolio (never modifies the working repo, never overwrites an existing folder).
Usage: python scripts/export_portfolio.py [--dest dist/portfolio] [--zip]
Excludes dataset text, model weights, embeddings, indexes, .env and review sheets; scrubs local paths; checks the result with the strict hygiene scan."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import find_repo_root
from src.export_tools import export


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=None)
    ap.add_argument("--zip", action="store_true")
    a = ap.parse_args()
    root = find_repo_root()
    spec = yaml.safe_load((root / "configs" / "export.yaml").read_text(encoding="utf-8"))["export"]
    dest = Path(a.dest) if a.dest else root / spec["default_dest"]
    dest = dest if dest.is_absolute() else root / dest
    m = export(root, dest, spec, make_zip=a.zip)
    print(f"Exported {m['n_files']} files to {dest.relative_to(root) if dest.is_relative_to(root) else dest}")
    print(f"  local-path scrubbing: {m['scrubbed_files']} files / {m['scrub_replacements']} replacements | strict hygiene scan: {m['hygiene_status']} {m['hygiene_counts'] or ''}")
    if not any(f["path"].upper().startswith("LICENSE") for f in m["files"]):
        print("  NOTE: no LICENSE file was found; choose a license for your code before publishing (see docs/PUBLISHING.md).")
    print("Next: review the folder, then follow docs/PUBLISHING.md (fresh repository; push only after your approval).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
