"""Phase 1: validate raw data, build processed corpus/queries, create the query-grouped split, write audit.

Usage:  python scripts/prepare_data.py [--config configs/base.yaml] [--force]
Derived files are staged first; existing files that would change are NOT overwritten without --force.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data_io import read_jsonl, read_qrels, write_qrels
from src.preprocess import build_processed_corpus, build_processed_queries
from src.split import partition_qrels, split_query_ids, verify_no_leakage
from src.utils import get_logger, set_seed, sha256_file, utc_now, write_json
from src.validate_data import (
    check_cross_qrels,
    validate_corpus,
    validate_qrels,
    validate_queries,
)

log = get_logger("prepare_data")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true", help="Allow overwriting changed derived files.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    raw, proc, reports = cfg["_raw"], cfg["_abs"]["processed_dir"], cfg["_abs"]["reports_dir"]

    # 1. Load + validate -------------------------------------------------------------------
    log.info("Loading raw files from %s", cfg["_abs"]["raw_dir"])
    corpus = read_jsonl(raw["corpus"])
    queries = read_jsonl(raw["queries"])
    train_rows = read_qrels(raw["train_qrels"])
    test_rows = read_qrels(raw["test_qrels"])

    corpus_audit = validate_corpus(corpus)
    query_audit = validate_queries(queries)
    corpus_ids = {r["_id"] for r in corpus}
    query_ids = {r["_id"] for r in queries}
    train_audit = validate_qrels("train.tsv", train_rows, corpus_ids, query_ids)
    test_audit = validate_qrels("test.tsv", test_rows, corpus_ids, query_ids)
    cross = check_cross_qrels(train_rows, test_rows)

    # 2. Split -----------------------------------------------------------------------------
    official_train_q = sorted({q for q, _, _ in train_rows})
    test_q = sorted({q for q, _, _ in test_rows})
    train_q, val_q = split_query_ids(official_train_q, cfg["split"]["val_fraction"], cfg["split"]["seed"])
    tr_rows, va_rows = partition_qrels(train_rows, train_q, val_q)
    gates = verify_no_leakage(train_q, val_q, test_q, tr_rows, va_rows)

    reconcile = {
        "queries_jsonl_total": len(query_ids),
        "official_train_queries": len(official_train_q),
        "official_test_queries": len(test_q),
        "train_plus_test_equals_queries_jsonl": len(official_train_q) + len(test_q) == len(query_ids),
        "queries_without_any_qrel": len(query_ids - set(official_train_q) - set(test_q)),
        "split_train_queries": len(train_q),
        "split_val_queries": len(val_q),
        "split_train_rows": len(tr_rows),
        "split_val_rows": len(va_rows),
        "split_rows_sum_equals_official_train_rows": len(tr_rows) + len(va_rows) == len(train_rows),
        "split_queries_sum_equals_official_train_queries": len(train_q) + len(val_q) == len(official_train_q),
    }
    all_pass = all(gates.values()) and reconcile["split_rows_sum_equals_official_train_rows"] \
        and reconcile["split_queries_sum_equals_official_train_queries"]

    # 3. Stage derived outputs, then install (never silently overwrite) ---------------------
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        sep = cfg["preprocess"]["title_text_separator"]
        pcorpus = build_processed_corpus(corpus, sep)
        pqueries = build_processed_queries(queries)
        with (stage / "corpus_processed.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
            for r in pcorpus:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        with (stage / "queries_processed.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
            for r in pqueries:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        write_json([r["_id"] for r in pcorpus], stage / "document_ids.json")
        write_json(train_q, stage / "train_query_ids.json")
        write_json(val_q, stage / "val_query_ids.json")
        write_json(test_q, stage / "test_query_ids.json")
        write_qrels(tr_rows, stage / "train_qrels.tsv")
        write_qrels(va_rows, stage / "val_qrels.tsv")

        conflicts = [
            f.name for f in stage.iterdir()
            if (proc / f.name).exists() and sha256_file(proc / f.name) != sha256_file(f)
        ]
        if conflicts and not args.force:
            log.error("Existing derived files differ from regenerated ones: %s. "
                      "Review, then re-run with --force to overwrite.", conflicts)
            return 2
        proc.mkdir(parents=True, exist_ok=True)
        status = {}
        for f in sorted(stage.iterdir()):
            dest = proc / f.name
            status[f.name] = "unchanged" if dest.exists() and sha256_file(dest) == sha256_file(f) else (
                "overwritten" if dest.exists() else "created")
            if status[f.name] != "unchanged":
                shutil.copyfile(f, dest)

    # 4. Over-length heuristic (true truncation is measured with the tokenizer in Phase 2) --
    budget = cfg["preprocess"]["approx_words_per_wordpiece_budget"]
    words = [len(r["doc_text"].split()) for r in pcorpus]
    n_long = sum(w > budget for w in words)
    truncation_note = {
        "max_seq_length_wordpieces": cfg["preprocess"]["max_seq_length"],
        "heuristic_word_budget": budget,
        "docs_exceeding_heuristic_budget": n_long,
        "fraction": round(n_long / len(words), 4),
        "caveat": "Whitespace-word heuristic only; exact truncation will be measured with the MiniLM "
                  "tokenizer in Phase 2. No text is truncated in the processed files.",
    }

    # 5. Reports ----------------------------------------------------------------------------
    audit = {
        "generated_utc": utc_now(),
        "status": "PASS" if all_pass else "FAIL",
        "seed": cfg["project"]["seed"],
        "val_fraction": cfg["split"]["val_fraction"],
        "raw_file_sha256": {k: sha256_file(p) for k, p in raw.items()},
        "corpus": corpus_audit,
        "queries": query_audit,
        "train_qrels_official": train_audit,
        "test_qrels_official": test_audit,
        "cross_checks": cross,
        "reconciliation": reconcile,
        "leakage_gates": gates,
        "document_ordering": "lexicographic by string _id (matches document_ids.json)",
        "truncation_heuristic": truncation_note,
        "processed_file_status": status,
        "notes": [
            "All qrel scores are 1 (positive-only); unjudged pairs are unlabeled, not verified negatives.",
            "Query metadata (SUPPORT/CONTRADICT annotations) is not a relevance label and is dropped from processed queries.",
            "test.tsv is used only to enumerate reserved test query IDs; its judgments are not copied or inspected further.",
        ],
    }
    write_json(audit, reports / "data_audit.json")
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    (reports / "environment_freeze.txt").write_text(
        f"# python {platform.python_version()} on {platform.platform()}\n{freeze}", encoding="utf-8")

    log.info("Status: %s | files: %s", audit["status"], status)
    log.info("Reconciliation: %s", reconcile)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
