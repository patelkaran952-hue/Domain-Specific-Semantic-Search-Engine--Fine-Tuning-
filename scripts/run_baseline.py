"""Phase 2: baseline (un-fine-tuned) MiniLM retrieval over the full corpus, evaluated on VALIDATION queries only.

Usage (PowerShell, repo root):
    python scripts/run_baseline.py --smoke     # tiny subset, no metrics: verifies the pipeline end to end
    python scripts/run_baseline.py             # full run: encode 5,183 docs, build FAISS index, evaluate val
    python scripts/run_baseline.py --force     # recompute metrics even if a report already exists
test.tsv is never read by this script.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data_io import read_jsonl, read_qrels
from src.embed import (
    CacheMismatch,
    cache_key,
    encode_texts,
    load_embeddings,
    load_sentence_model,
    make_encode_fn,
    measure_truncation,
    save_embeddings,
)
from src.evaluate import METRIC_DEFINITIONS, evaluate_run, qrels_to_dict
from src.index_faiss import build_index, check_unit_norm, load_index, save_index, search
from src.utils import get_logger, read_json, set_seed, sha256_file, utc_now, write_json

log = get_logger("run_baseline")
_PROC = psutil.Process()
_peak = {"mb": 0.0}


def rss_mb() -> float:
    cur = _PROC.memory_info().rss / 1e6
    _peak["mb"] = max(_peak["mb"], cur)
    return cur


def need(path: Path, hint: str) -> Path:
    if not path.is_file():
        raise SystemExit(f"Missing {path}. {hint}")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--rebuild-embeddings",
        action="store_true",
        help="ignore a valid embedding cache",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = cfg["project"]["seed"]
    set_seed(seed)
    proc, reports = cfg["_abs"]["processed_dir"], cfg["_abs"]["reports_dir"]
    root = cfg["_abs"]["embeddings_dir"].parent
    if args.smoke:
        emb_dir, idx_dir, reports = root / "smoke", root / "smoke", root / "smoke"
    else:
        emb_dir, idx_dir = cfg["_abs"]["embeddings_dir"], cfg["_abs"]["indexes_dir"]
    report_path = reports / "baseline_val_metrics.json"
    if report_path.exists() and not args.smoke and not args.force:
        raise SystemExit(
            f"{report_path} already exists. Review it, or pass --force to recompute (old file is replaced)."
        )

    # ---- inputs (validated) ---------------------------------------------------------------
    hint = "Run `python scripts/prepare_data.py` (Phase 1) first."
    corpus_p = need(proc / "corpus_processed.jsonl", hint)
    ids_p = need(proc / "document_ids.json", hint)
    queries_p = need(proc / "queries_processed.jsonl", hint)
    val_ids_p = need(proc / "val_query_ids.json", hint)
    val_qrels_p = need(proc / "val_qrels.tsv", hint)
    docs = read_jsonl(corpus_p)
    doc_ids = read_json(ids_p)
    if [d["_id"] for d in docs] != doc_ids:
        raise SystemExit(
            "corpus_processed.jsonl order differs from document_ids.json; re-run prepare_data.py."
        )
    queries = {r["_id"]: r["text"] for r in read_jsonl(queries_p)}
    val_ids = read_json(val_ids_p)
    qrels = qrels_to_dict(read_qrels(val_qrels_p))
    if set(qrels) != set(val_ids):
        raise SystemExit("val_qrels.tsv query IDs do not match val_query_ids.json.")
    corpus_hash = sha256_file(corpus_p)
    ec, mc = cfg["embedding"], cfg["model"]

    if args.smoke:
        docs, doc_ids = (
            docs[: cfg["smoke"]["n_docs"]],
            doc_ids[: cfg["smoke"]["n_docs"]],
        )
        val_ids = val_ids[: cfg["smoke"]["n_queries"]]
        log.info(
            "SMOKE MODE: %d docs, %d queries, no metrics will be reported.",
            len(docs),
            len(val_ids),
        )

    # ---- model -----------------------------------------------------------------------------
    t0 = time.perf_counter()
    model, prov = load_sentence_model(mc["name"], mc["revision"], mc["max_seq_length"])
    if prov["embedding_dim"] != mc["embedding_dim"]:
        raise SystemExit(
            f"Model dim {prov['embedding_dim']} != configured {mc['embedding_dim']}."
        )
    model_load_s = time.perf_counter() - t0
    encode_fn = make_encode_fn(model, ec["batch_size"])
    rss_mb()
    texts = [d["doc_text"] for d in docs]
    trunc = measure_truncation(model.tokenizer, texts, mc["max_seq_length"])
    log.info("Truncation: %s", trunc)

    # ---- document embeddings (cached) -----------------------------------------------------
    stem = "baseline_docs" if not args.smoke else "smoke_docs"
    key = cache_key(
        corpus_hash + (":smoke" if args.smoke else ""),
        prov,
        ec["normalize"],
        ec["batch_size"],
    )
    cache_used = False
    try:
        if args.rebuild_embeddings:
            raise FileNotFoundError("rebuild requested")
        doc_emb = load_embeddings(emb_dir, stem, key)
        cache_used = True
        encode_docs_s = 0.0
        log.info("Reusing cached document embeddings %s", doc_emb.shape)
    except (FileNotFoundError, CacheMismatch) as exc:
        log.info("Encoding documents (%s)", exc)
        t0 = time.perf_counter()
        doc_emb = encode_texts(encode_fn, texts, ec["chunk_size"], ec["normalize"])
        encode_docs_s = time.perf_counter() - t0
        save_embeddings(
            doc_emb,
            key,
            {"seed": seed, "batch_size": ec["batch_size"], "provenance": prov},
            emb_dir,
            stem,
        )
    rss_mb()
    if (
        doc_emb.shape != (len(docs), prov["embedding_dim"])
        or doc_emb.dtype != np.float32
    ):
        raise SystemExit(
            f"Unexpected embedding matrix {doc_emb.shape} {doc_emb.dtype}."
        )
    norm_dev = check_unit_norm(doc_emb, cfg["checks"]["norm_tolerance"])

    # ---- index -----------------------------------------------------------------------------
    index = build_index(doc_emb, cfg["checks"]["norm_tolerance"])
    idx_stem = "baseline_flatip" if not args.smoke else "smoke_flatip"
    idx_meta = save_index(
        index,
        doc_ids,
        idx_dir,
        idx_stem,
        {
            "model": prov,
            "corpus_sha256": corpus_hash,
            "index_type": "IndexFlatIP",
            "normalized": True,
        },
    )
    index2, ids2 = load_index(
        idx_dir, idx_stem, expected_doc_ids=doc_ids
    )  # reload from disk

    # ---- queries + retrieval --------------------------------------------------------------
    q_texts = [queries[q] for q in val_ids]
    t0 = time.perf_counter()
    q_emb = encode_texts(encode_fn, q_texts, ec["chunk_size"], ec["normalize"])
    encode_q_s = time.perf_counter() - t0
    top_k = cfg["retrieval"]["top_k"]
    results = search(index2, ids2, q_emb, top_k)
    run = {q: [d for d, _ in r] for q, r in zip(val_ids, results)}
    rss_mb()

    # ---- consistency checks ----------------------------------------------------------------
    from sklearn.metrics.pairwise import cosine_similarity

    rng = np.random.default_rng(seed)
    n = min(cfg["checks"]["n_consistency_samples"], len(val_ids))
    ci = rng.choice(len(val_ids), size=n, replace=False)
    ip_vs_np = ip_vs_cos = 0.0
    for i in ci:
        ref_full = cosine_similarity(q_emb[i : i + 1], doc_emb)[0]
        for doc, score in results[i][:10]:
            j = ids2.index(doc)
            ip_vs_np = max(ip_vs_np, abs(score - float(q_emb[i] @ doc_emb[j])))
            ip_vs_cos = max(ip_vs_cos, abs(score - float(ref_full[j])))
    same_as_memory = search(index, doc_ids, q_emb, top_k) == results
    rs = min(cfg["checks"]["reproducibility_resample_docs"], len(docs))
    pick = rng.choice(len(docs), size=rs, replace=False)
    re_emb = encode_texts(encode_fn, [texts[i] for i in pick], ec["chunk_size"], True)
    repro_dev = float(np.max(np.abs(re_emb - doc_emb[pick])))
    ordering_ok = ids2 == doc_ids and idx_meta["ntotal"] == len(doc_ids)
    sorted_ok = all(
        [s for _, s in r] == sorted((s for _, s in r), reverse=True) for r in results
    )
    checks = {
        "doc_index_mapping_consistent": ordering_ok,
        "reloaded_index_equals_in_memory_results": same_as_memory,
        "scores_sorted_descending": sorted_ok,
        "unit_norm_max_deviation": norm_dev,
        "ip_vs_numpy_dot_max_abs_diff": ip_vs_np,
        "ip_vs_sklearn_cosine_max_abs_diff": ip_vs_cos,
        "ip_equals_cosine": ip_vs_np <= cfg["checks"]["score_tolerance"]
        and ip_vs_cos <= cfg["checks"]["score_tolerance"],
        "reencode_sample_max_abs_diff_vs_cache": repro_dev,
        "reencode_reproduces_cache": repro_dev
        <= cfg["checks"]["reproducibility_tolerance"],
    }
    bool_ok = all(v for k, v in checks.items() if isinstance(v, bool))
    log.info("Checks: %s", checks)

    if args.smoke:
        print(
            f"\nSMOKE {'PASS' if bool_ok else 'FAIL'}: docs={doc_emb.shape}, index.ntotal={index.ntotal}, "
            f"queries={len(val_ids)}, top-1 scores={[round(r[0][1], 3) for r in results]}\nchecks={checks}"
        )
        return 0 if bool_ok else 1

    # ---- evaluation (validation only) -------------------------------------------------------
    ev = cfg["evaluation"]
    res = evaluate_run(
        run,
        qrels,
        ks=tuple(ev["ks"]),
        ndcg_k=ev["ndcg_k"],
        mrr_k=ev["mrr_k"],
        precision_k=ev["precision_k"],
        map_k=ev["map_k"],
    )
    res2 = evaluate_run(
        {
            q: [d for d, _ in r]
            for q, r in zip(val_ids, search(index2, ids2, q_emb, top_k))
        },
        qrels,
        ks=tuple(ev["ks"]),
        ndcg_k=ev["ndcg_k"],
        mrr_k=ev["mrr_k"],
        precision_k=ev["precision_k"],
        map_k=ev["map_k"],
    )
    checks["baseline_metrics_reproduced_on_rerun"] = (
        res["aggregate"] == res2["aggregate"]
    )
    bool_ok = bool_ok and checks["baseline_metrics_reproduced_on_rerun"]

    # ---- latency (single query, after warm-up; startup reported separately) ------------------
    enc_ms, srch_ms = [], []
    for k_, q in enumerate(q_texts):
        t = time.perf_counter()
        e = encode_fn([q])
        t1 = time.perf_counter()
        e = e / max(np.linalg.norm(e), 1e-12)
        search(index2, ids2, e.astype(np.float32), top_k)
        t2 = time.perf_counter()
        if k_ >= 3:
            enc_ms.append((t1 - t) * 1e3)
            srch_ms.append((t2 - t1) * 1e3)
    pct = lambda a, p: float(np.percentile(a, p))
    latency = {
        "n_timed_queries": len(enc_ms),
        "encode_ms": {
            "mean": statistics.fmean(enc_ms),
            "p50": pct(enc_ms, 50),
            "p95": pct(enc_ms, 95),
        },
        "faiss_search_ms": {
            "mean": statistics.fmean(srch_ms),
            "p50": pct(srch_ms, 50),
            "p95": pct(srch_ms, 95),
        },
    }
    rss_mb()

    report = {
        "run_id": "baseline_minilm_val",
        "generated_utc": utc_now(),
        "status": "PASS" if bool_ok else "FAIL",
        "split": "validation",
        "seed": seed,
        "eval_setup": {
            "n_queries": res["n_queries"],
            "n_judgments": res["n_judgments"],
            "candidate_corpus_size": index.ntotal,
            "retrieval_top_k": top_k,
            "queries_missing_from_run": res["queries_missing_from_run"],
            "note": "Metrics compare against sparse known positives, not exhaustive relevance.",
        },
        "metrics": res["aggregate"],
        "metric_definitions": METRIC_DEFINITIONS,
        "model": prov,
        "embedding": {
            **ec,
            "corpus_sha256": corpus_hash,
            "cache_reused": cache_used,
            "embeddings_sha256": sha256_file(emb_dir / f"{stem}.npy"),
        },
        "index": {
            "type": "IndexFlatIP",
            "ntotal": index.ntotal,
            "dim": index.d,
            "index_sha256": idx_meta["index_sha256"],
        },
        "truncation": trunc,
        "checks": checks,
        "timing_s": {
            "model_load": model_load_s,
            "encode_documents": encode_docs_s,
            "encode_val_queries_batch": encode_q_s,
        },
        "latency_single_query": latency,
        "memory_mb": {
            "peak_rss_observed": round(_peak["mb"], 1),
            "note": "RSS sampled at stage boundaries, not a true peak.",
        },
        "environment": {"platform": sys.platform, "python": sys.version.split()[0]},
    }
    write_json(report, report_path)
    write_json(
        {q: [[d, s] for d, s in r] for q, r in zip(val_ids, results)},
        reports / "baseline_val_run_top100.json",
    )
    cols = list(next(iter(res["per_query"].values())).keys())
    with (reports / "baseline_val_per_query.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        w = csv.writer(fh)
        w.writerow(["query_id", *cols])
        for q in val_ids:
            w.writerow([q, *[f"{res['per_query'][q][c]:.6f}" for c in cols]])
    runs_csv = cfg["_root"] / "experiments" / "runs.csv"
    new = not runs_csv.exists()
    with runs_csv.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(
                [
                    "timestamp_utc",
                    "phase",
                    "run_id",
                    "model",
                    "revision",
                    "split",
                    "n_queries",
                    *cols,
                    "seed",
                    "status",
                ]
            )
        w.writerow(
            [
                report["generated_utc"],
                2,
                report["run_id"],
                mc["name"],
                prov["resolved_revision"],
                "val",
                res["n_queries"],
                *[f"{res['aggregate'][c]:.6f}" for c in cols],
                seed,
                report["status"],
            ]
        )

    print("\n=== Baseline (validation) ===")
    for k_, v in res["aggregate"].items():
        print(f"{k_:>14}: {v:.4f}")
    print(
        f"queries={res['n_queries']} judgments={res['n_judgments']} corpus={index.ntotal} top_k={top_k}"
    )
    print(f"status={report['status']} | checks={checks}")
    return 0 if bool_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
