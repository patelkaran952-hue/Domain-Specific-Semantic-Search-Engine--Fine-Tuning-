"""Verify that all Phase 2 dependencies import on THIS machine and (optionally) that the model can be fetched.
Usage: python scripts/check_env.py [--download-model]"""
from __future__ import annotations

import argparse
import importlib
import platform
import sys

PKGS = ["numpy", "pandas", "yaml", "sklearn", "torch", "transformers", "sentence_transformers",
        "huggingface_hub", "faiss", "psutil", "pytest"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-model", action="store_true", help="also fetch all-MiniLM-L6-v2 (about 90 MB)")
    a = ap.parse_args()
    print(f"python {platform.python_version()} | {platform.platform()}")
    bad = []
    for name in PKGS:
        try:
            m = importlib.import_module(name)
            print(f"  OK   {name:<22} {getattr(m, '__version__', '?')}")
        except Exception as exc:  # noqa: BLE001
            bad.append(name)
            print(f"  FAIL {name:<22} {type(exc).__name__}: {exc}")
    if a.download_model and not bad:
        from huggingface_hub import snapshot_download

        path = snapshot_download("sentence-transformers/all-MiniLM-L6-v2")
        print("  model snapshot:", path)
    if bad:
        print(f"\nMissing/broken: {bad}. Install with `python -m pip install -r requirements.txt` "
              "(faiss-cpu wheels are not available for every Python version; use Python 3.11 if it fails).")
        return 1
    print("\nEnvironment OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
