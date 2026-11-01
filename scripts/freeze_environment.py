"""Pin the direct dependencies to the versions INSTALLED in this environment (the ones the results were produced with).
Usage: python scripts/freeze_environment.py [--force] [--licenses] [--full]
Writes requirements.lock.txt (and with --full requirements.lock.full.txt = every installed distribution). Built from importlib.metadata,
not `pip freeze`, so conda `@ file:///...` paths cannot leak into the file."""
from __future__ import annotations

import argparse
import sys
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import find_repo_root
from src.envtools import license_info, norm, parse_requirements, render_lock


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--licenses", action="store_true", help="print the license metadata of each direct dependency")
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    root = find_repo_root()
    names = sorted({n for f in ("requirements.txt", "requirements-dev.txt") if (root / f).is_file() for n, _ in parse_requirements(root / f)})
    if a.licenses:
        print(f"{'package':<26}{'installed':<14}license metadata (as declared by the package)")
        for n in names:
            try:
                v = metadata.version(n)
            except metadata.PackageNotFoundError:
                v = "-"
            print(f"{n:<26}{v:<14}{license_info(n)}")
    out = root / "requirements.lock.txt"
    if out.exists() and not a.force:
        print(f"{out.name} already exists; re-run with --force to replace it.")
        return 2
    out.write_text(render_lock(names), encoding="utf-8", newline="\n")
    print(f"wrote {out.name}:\n{out.read_text(encoding='utf-8')}")
    if a.full:
        full = root / "requirements.lock.full.txt"
        if full.exists() and not a.force:
            print(f"{full.name} exists; use --force")
            return 2
        dists = sorted({(norm(d.metadata["Name"]), d.version) for d in metadata.distributions() if d.metadata["Name"]})
        full.write_text("\n".join(f"{n}=={v}" for n, v in dists) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {full.name} ({len(dists)} distributions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
