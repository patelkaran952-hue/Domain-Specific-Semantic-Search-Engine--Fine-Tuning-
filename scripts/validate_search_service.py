"""Phase 6: validate the built package (fresh reload vs Phase 5 rankings, input handling, cosine check) and benchmark latency
after startup. Usage: python scripts/validate_search_service.py [--force]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.service_validation import run_service_validation


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="replace an existing search_service_report.json")
    a = ap.parse_args()
    cfg = load_config(["configs/baseline.yaml", "configs/compare.yaml", "configs/service.yaml"])
    pkg = cfg["_abs"]["service_dir"] / cfg["service"]["package_name"]
    rp = cfg["_abs"]["reports_dir"] / "search_service_report.json"
    if not (pkg / "service_manifest.json").is_file():
        raise SystemExit(f"{pkg} not found. Run `python scripts/build_search_package.py` first.")
    if rp.exists() and not a.force:
        raise SystemExit(f"{rp.name} already exists. Re-run with --force to replace it.")
    r = run_service_validation(cfg, pkg)
    rc = r["reload_consistency"]
    print(f"\n=== Search service validation: {r['status']} ===")
    print(f"reload vs Phase 5 rankings ({rc['n_queries']} val queries, depth {rc['compared_depth']}): exact={rc['exact']} "
          f"tie-aware-equivalent={rc['tie_aware_equivalent_not_exact']} not-equivalent={rc['not_equivalent']}")
    e = r["end_to_end_metric"]
    print(f"service val nDCG@10 {e['service_val_ndcg@10']:.4f} vs Phase 5 {e['phase5_val_ndcg@10']:.4f} (abs diff {e['abs_diff']:.2e})")
    print(f"startup (excluded from latency): { {k: round(v, 2) for k, v in r['startup_s_excluded_from_latency'].items()} }")
    for k, b in r["latency_after_startup"].items():
        t = b["total_ms"]
        print(f"  {k:<10} total ms: mean {t['mean']:.1f} p50 {t['p50']:.1f} p95 {t['p95']:.1f} p99 {t['p99']:.1f} | encode {b['encode_ms_mean']:.1f} faiss {b['faiss_search_ms_mean']:.2f} | {b['throughput_qps_sequential']:.1f} q/s")
    print(f"memory: {r['memory_mb']} | package: {r['package']['size_mb']} MB in {r['package']['n_files']} files")
    for q, hits in r["example_queries"].items():
        print(f"\nExample: {q}")
        for h in hits:
            print(f"  #{h['rank']} [{h['corpus_id']}] cos={h['similarity_score']:.3f}  {h['title'][:90]}")
    print("\nfailed gates:", [k for k, v in r["gates"].items() if not v] or "none")
    return 0 if r["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
