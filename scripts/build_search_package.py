"""Phase 6: package the FROZEN selected model (weights, normalized embeddings, exact FAISS index, ID map, documents) into one
hash-verified directory. Never overwrites an existing package. Usage: python scripts/build_search_package.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.search_service import PackagingError, build_package


def main() -> int:
    cfg = load_config(["configs/baseline.yaml", "configs/compare.yaml", "configs/service.yaml"])
    try:
        m = build_package(cfg)
    except PackagingError as exc:
        print(f"PACKAGING BLOCKED: {exc}")
        return 4
    print(f"Built {cfg['_abs']['service_dir'] / cfg['service']['package_name']}")
    print(f"  model={m['selected_model']} docs={m['n_documents']} dim={m['embedding_dim']} files={len(m['files'])} "
          f"size={sum(v['size'] for v in m['files'].values()) / 1e6:.1f} MB")
    print(f"  freeze fingerprint {m['freeze_fingerprint'][:16]}… | reported metrics: {m['reported_metrics']['heldout_test'].get('ndcg@10')} (held-out test nDCG@10, from Phase 5)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
