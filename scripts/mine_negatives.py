"""Phase 3: build positive pairs and mine hard negatives for TRAINING queries only.

Usage (PowerShell, repo root):
    python scripts/mine_negatives.py --source bm25     # lexical; needs only the processed data
    python scripts/mine_negatives.py --source dense    # baseline MiniLM; needs the Phase 2 embeddings (run_baseline.py first)
    python scripts/mine_negatives.py                   # all sources listed in configs/mining.yaml
Existing files that would change are never overwritten without --force. val_qrels.tsv / test.tsv are never read.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data_io import read_jsonl, read_qrels
from src.negative_mining import (BM25, build_positive_pairs, dense_score_matrix, mine_query, mining_stats,
                                 read_jsonl_rows, sample_triplets, verify_mining_outputs, write_jsonl, write_samples_md)
from src.utils import get_logger, install_staged, read_json, sha256_file, utc_now, write_json

log = get_logger("mine_negatives")


def need(path: Path, hint: str) -> Path:
    if not path.is_file():
        raise SystemExit(f"Missing {path}. {hint}")
    return path


def dense_scores(cfg, docs, doc_ids, q_texts, corpus_hash):
    """Baseline-MiniLM scores using the Phase 2 cached embeddings and (as a cross-check) its FAISS index."""
    from src.embed import CacheMismatch, cache_key, encode_texts, load_embeddings, load_sentence_model, make_encode_fn
    from src.index_faiss import load_index, search

    base = need(cfg["_abs"]["reports_dir"] / "baseline_val_metrics.json", "Run `python scripts/run_baseline.py` (Phase 2) first.")
    base_report = read_json(base)
    if base_report.get("status") != "PASS":
        raise SystemExit(f"{base} has status {base_report.get('status')!r}; fix Phase 2 before dense mining.")
    mc, ec = cfg["model"], cfg["embedding"]
    model, prov = load_sentence_model(mc["name"], mc["revision"], mc["max_seq_length"])
    key = cache_key(corpus_hash, prov, ec["normalize"], ec["batch_size"])
    try:
        doc_emb = load_embeddings(cfg["_abs"]["embeddings_dir"], "baseline_docs", key)
    except (FileNotFoundError, CacheMismatch) as exc:
        raise SystemExit(f"Cannot reuse the Phase 2 embeddings ({exc}). Re-run scripts/run_baseline.py; "
                         "dense mining will not silently re-encode with a different model.") from exc
    q_emb = encode_texts(make_encode_fn(model, ec["batch_size"]), q_texts, ec["chunk_size"], ec["normalize"])
    S = dense_score_matrix(q_emb, doc_emb)
    index, ids = load_index(cfg["_abs"]["indexes_dir"], "baseline_flatip", expected_doc_ids=doc_ids)
    sample = np.random.default_rng(cfg["project"]["seed"]).choice(len(q_texts), size=min(25, len(q_texts)), replace=False)
    faiss_rows = search(index, ids, q_emb[sample], 10)
    agree = all([d for d, _ in row] == [doc_ids[j] for j in np.lexsort((np.arange(S.shape[1]), -S[i]))[:10]]
                for row, i in zip(faiss_rows, sample))
    extra = {"model": prov, "embeddings_sha256": sha256_file(cfg["_abs"]["embeddings_dir"] / "baseline_docs.npy"),
             "baseline_report_sha256": sha256_file(base), "dense_top10_matches_faiss_index_on_sample": bool(agree)}
    return S, extra


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", default=["configs/baseline.yaml", "configs/mining.yaml"])
    ap.add_argument("--source", choices=["bm25", "dense"], action="append")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    mc, seed = cfg["mining"], cfg["project"]["seed"]
    sources = args.source or mc["sources"]
    proc, mdir, reports = cfg["_abs"]["processed_dir"], cfg["_abs"]["mining_dir"], cfg["_abs"]["reports_dir"]

    hint = "Run `python scripts/prepare_data.py` (Phase 1) first."
    corpus_p = need(proc / "corpus_processed.jsonl", hint)
    train_p = need(proc / "train_qrels.tsv", hint)
    docs = read_jsonl(corpus_p)
    doc_ids = [d["_id"] for d in docs]
    if doc_ids != read_json(need(proc / "document_ids.json", hint)):
        raise SystemExit("corpus_processed.jsonl order differs from document_ids.json; re-run prepare_data.py.")
    queries = {r["_id"]: r["text"] for r in read_jsonl(need(proc / "queries_processed.jsonl", hint))}
    titles = {d["_id"]: d["title"] for d in docs}
    train_ids = read_json(need(proc / "train_query_ids.json", hint))
    val_ids = read_json(need(proc / "val_query_ids.json", hint))
    test_ids = read_json(need(proc / "test_query_ids.json", hint))
    train_rows = read_qrels(train_p)                   # TRAINING qrels only
    if {q for q, _, _ in train_rows} != set(train_ids):
        raise SystemExit("train_qrels.tsv queries differ from train_query_ids.json.")
    corpus_hash = sha256_file(corpus_p)
    pairs = build_positive_pairs(train_rows)
    positives: dict[str, set[str]] = {}
    for q, d, s in train_rows:
        positives.setdefault(q, set()).add(d)
    q_texts = [queries[q] for q in train_ids]
    log.info("%d training queries, %d positive pairs, sources=%s", len(train_ids), len(pairs), sources)

    exit_code = 0
    for source in sources:
        t0 = time.perf_counter()
        extra: dict = {}
        if source == "bm25":
            S = BM25(["" + d["doc_text"] for d in docs], mc["bm25"]["k1"], mc["bm25"]["b"],
                     mc["bm25"]["token_pattern"]).score_matrix(q_texts)
        else:
            S, extra = dense_scores(cfg, docs, doc_ids, q_texts, corpus_hash)
        mined = [mine_query(q, positives[q], S[i], doc_ids, candidate_depth=mc["candidate_depth"], skip_top=mc["skip_top"],
                            pool_size=mc["pool_size"], min_score=mc["min_score"], max_score_ratio=mc["max_score_ratio"],
                            flag_score_ratio=mc["flag_score_ratio"]) for i, q in enumerate(train_ids)]
        mined_by_q = {m["query_id"]: m for m in mined}
        triplets = sample_triplets(pairs, mined_by_q, seed, mc["triplets_per_pair"])
        settings = {"source": source, "seed": seed, "mining": {k: v for k, v in mc.items() if k not in ("sources",)},
                    "inputs_sha256": {"corpus_processed.jsonl": corpus_hash, "train_qrels.tsv": sha256_file(train_p),
                                      "train_query_ids.json": sha256_file(proc / "train_query_ids.json")},
                    "provenance": extra,
                    "warning": "Negatives are unjudged top-ranked documents; some are likely false negatives."}
        with tempfile.TemporaryDirectory() as tmp:
            st = Path(tmp)
            write_jsonl(pairs, st / "train_positive_pairs.jsonl")
            write_jsonl(mined, st / f"hard_negatives_{source}.jsonl")
            write_jsonl(triplets, st / f"train_triplets_{source}.jsonl")
            write_json(settings, st / f"mining_settings_{source}.json")
            write_samples_md(st / f"samples_{source}.md", mined, queries, titles, mc["n_samples_for_inspection"], seed)
            try:
                status_files = install_staged(st, mdir, args.force)
            except FileExistsError as exc:
                log.error("%s", exc)
                return 2

        gates = verify_mining_outputs(mdir, source, train_rows, train_ids, val_ids, test_ids, set(doc_ids), mc["pool_size"])
        if "dense_top10_matches_faiss_index_on_sample" in extra:
            gates["dense_top10_matches_faiss_index_on_sample"] = extra["dense_top10_matches_faiss_index_on_sample"]
        ok = all(gates.values())
        stats = mining_stats(mined, train_rows, mc["pool_size"], mc["flag_score_ratio"])
        report = {"source": source, "status": "PASS" if ok else "FAIL", "generated_utc": utc_now(),
                  "elapsed_s": round(time.perf_counter() - t0, 2), "gates": gates, "stats": stats,
                  "n_positive_pairs": len(pairs), "n_triplets": len(triplets), "file_status": status_files,
                  "output_sha256": {n: sha256_file(mdir / n) for n in status_files},
                  "caveats": ["Unjudged top-ranked documents are not verified negatives.",
                              "Training-query diagnostics here are NOT validation metrics.",
                              "Only training-partition qrels were read; val/test judgments were not."]}
        write_json(report, reports / f"negative_mining_{source}_report.json")
        print(f"\n=== {source}: {report['status']} | pairs={len(pairs)} triplets={len(triplets)} | {status_files}")
        print("gates:", gates)
        print("negative score:", stats["negative_score"])
        print("flagged false-negative-risk:", stats["flagged_false_negative_risk"])
        print("train best-positive rank diagnostics:", stats["train_query_best_positive_rank_diagnostics"])
        exit_code = exit_code or (0 if ok else 1)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
