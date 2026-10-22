"""Phase 1 smoke test: confirms processed artifacts exist and are mutually consistent.
Usage: python scripts/smoke_test.py   (exit code 0 = PASS)"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data_io import read_jsonl, read_qrels
from src.split import verify_no_leakage
from src.utils import read_json


def main() -> int:
    cfg = load_config()
    proc = cfg["_abs"]["processed_dir"]
    need = ["corpus_processed.jsonl", "queries_processed.jsonl", "document_ids.json", "train_query_ids.json",
            "val_query_ids.json", "test_query_ids.json", "train_qrels.tsv", "val_qrels.tsv"]
    missing = [n for n in need if not (proc / n).is_file()]
    if missing:
        print(f"FAIL: missing {missing}. Run `python scripts/prepare_data.py`.")
        return 1
    docs = read_jsonl(proc / "corpus_processed.jsonl")
    doc_ids = read_json(proc / "document_ids.json")
    checks = {
        "document_ids matches corpus order": [d["_id"] for d in docs] == doc_ids,
        "document_ids sorted & unique": doc_ids == sorted(set(doc_ids)),
    }
    tr, va = read_qrels(proc / "train_qrels.tsv"), read_qrels(proc / "val_qrels.tsv")
    tq, vq, sq = (read_json(proc / f"{n}_query_ids.json") for n in ("train", "val", "test"))
    checks.update({f"gate: {k}": v for k, v in verify_no_leakage(tq, vq, sq, tr, va).items()})
    qids = {r["_id"] for r in read_jsonl(proc / "queries_processed.jsonl")}
    ids = set(doc_ids)
    checks["all qrel IDs resolve"] = all(q in qids and d in ids for q, d, _ in tr + va) and set(sq) <= qids
    for k, v in checks.items():
        print(("PASS" if v else "FAIL"), "-", k)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
