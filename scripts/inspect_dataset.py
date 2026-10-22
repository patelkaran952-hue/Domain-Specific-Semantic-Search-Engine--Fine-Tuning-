"""Phase 1: print dataset statistics and run a few training-free sample searches.

Sample queries are drawn ONLY from the training partition (never validation or test).
Usage: python scripts/inspect_dataset.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data_io import read_jsonl, read_qrels
from src.retrieve import TfidfProbe
from src.utils import read_json, utc_now, write_json


def main() -> int:
    cfg = load_config()
    proc, reports = cfg["_abs"]["processed_dir"], cfg["_abs"]["reports_dir"]
    audit_path = reports / "data_audit.json"
    if not audit_path.is_file():
        raise SystemExit(f"{audit_path} not found. Run `python scripts/prepare_data.py` first.")
    audit = read_json(audit_path)

    print("=== Dataset audit summary ===")
    print(f"status: {audit['status']}")
    print(f"corpus docs: {audit['corpus']['n_documents']} | abstract words: {audit['corpus']['abstract_words']}")
    print(f"queries: {audit['queries']['n_queries']} | query words: {audit['queries']['query_words']}")
    print(f"reconciliation: {audit['reconciliation']}")

    docs = read_jsonl(proc / "corpus_processed.jsonl")
    queries = {r["_id"]: r["text"] for r in read_jsonl(proc / "queries_processed.jsonl")}
    train_ids = read_json(proc / "train_query_ids.json")
    qrels = defaultdict(set)
    for q, d, _ in read_qrels(proc / "train_qrels.tsv"):
        qrels[q].add(d)

    sc = cfg["sample_search"]
    rng = np.random.default_rng(sc["seed"])
    picks = [train_ids[i] for i in rng.choice(len(train_ids), size=sc["n_queries"], replace=False)]
    probe = TfidfProbe(docs)
    title_of = {d["_id"]: d["title"] for d in docs}

    samples = []
    print("\n=== Sample TF-IDF searches (training queries only; lexical sanity probe, not a baseline) ===")
    for qid in picks:
        hits = probe.search(queries[qid], sc["top_k"])
        found = [h["corpus_id"] for h in hits if h["corpus_id"] in qrels[qid]]
        print(f"\nQuery {qid}: {queries[qid]}")
        print(f"  judged positives: {sorted(qrels[qid])} -> {[title_of[d][:70] for d in sorted(qrels[qid])]}")
        for h in hits:
            mark = "*" if h["corpus_id"] in qrels[qid] else " "
            print(f"  {mark} #{h['rank']} [{h['corpus_id']}] {h['score']:.3f}  {h['title'][:80]}")
        samples.append({"query_id": qid, "query": queries[qid], "positives": sorted(qrels[qid]),
                        "top_k": hits, "positives_in_top_k": found})
    write_json({"generated_utc": utc_now(), "note": "TF-IDF sanity probe on training queries; not a reported baseline.",
                "samples": samples}, reports / "sample_searches.json")
    print(f"\nSaved {reports / 'sample_searches.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
