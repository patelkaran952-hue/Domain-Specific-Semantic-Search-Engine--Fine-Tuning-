"""Query the packaged index from the command line. Usage: python scripts/search_cli.py "claim or topic" --top-k 5"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.search_service import SearchService


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--no-verify", action="store_true", help="skip SHA-256 verification of the package at startup")
    a = ap.parse_args()
    cfg = load_config(["configs/baseline.yaml", "configs/compare.yaml", "configs/service.yaml"])
    svc = SearchService.load(cfg["_abs"]["service_dir"] / cfg["service"]["package_name"], verify=not a.no_verify)
    r = svc.search_detailed(a.query, a.top_k)
    print(f"[startup {svc.startup_s['total']:.1f}s | query {r.latency_ms.get('total', 0):.0f} ms | status={r.status}]  (cosine similarity, not a probability)")
    for w in r.warnings:
        print("warning:", w)
    for h in r.hits:
        print(f"\n#{h.rank}  [{h.corpus_id}]  cos={h.similarity_score:.3f}\n    {h.title}\n    {h.snippet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
