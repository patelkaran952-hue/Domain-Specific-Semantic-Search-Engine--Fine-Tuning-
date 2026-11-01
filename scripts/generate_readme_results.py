"""Fill the generated blocks of README.md and docs/MODEL_CARD.md from the MEASURED report files (no number is ever typed by hand).
Usage: python scripts/generate_readme_results.py [--reports-dir artifacts/reports] [--check]
--check exits 1 if the files are out of date (use before publishing)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import find_repo_root
from src.readme_results import load_reports, render_targets, replace_block


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reports-dir", default=None)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    root = find_repo_root()
    rd = Path(a.reports_dir) if a.reports_dir else next((p for p in (root / "artifacts" / "reports", root / "results" / "reports")
                                                          if (p / "test_results.json").is_file()), root / "artifacts" / "reports")
    rd = rd if rd.is_absolute() else root / rd
    R = load_reports(rd)
    stale = []
    for rel, blocks in render_targets(R, root).items():
        p = root / rel
        if not p.is_file():
            print(f"skip {rel}: file not found")
            continue
        old = text = p.read_text(encoding="utf-8")
        for name, body in blocks.items():
            try:
                text = replace_block(text, name, body)
            except ValueError as exc:
                print(f"skip block {name} in {rel}: {exc}")
        if text != old:
            stale.append(rel)
            if not a.check:
                p.write_text(text, encoding="utf-8", newline="\n")
    found = [k for k in ("audit", "val", "test", "service", "cal", "rag") if R[k]]
    print(f"reports read from {rd.relative_to(root) if rd.is_relative_to(root) else rd}: found {found}; fine-tune runs {list(R['finetune'])}; review summaries {list(R['review'])}")
    if a.check:
        print("OUT OF DATE: " + ", ".join(stale) if stale else "README/model card are up to date with the reports.")
        return 1 if stale else 0
    print("updated: " + ", ".join(stale) if stale else "no changes (already up to date).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
