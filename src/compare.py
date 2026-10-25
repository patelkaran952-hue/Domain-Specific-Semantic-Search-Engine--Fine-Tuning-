"""Phase 5: shared-implementation model comparison, validation-only selection, freezing, and the single held-out
test evaluation. Model loading is injected (`loader`), so all logic is testable without torch.

Protocol (enforced in code, not just documented):
  1. VALIDATION stage scores every candidate with src/evaluate.py on the same corpus/queries/depth, selects ONE model with a
     pre-registered rule, and writes a freeze record (weights hashes + config fingerprint).
  2. TEST stage runs only if the freeze is intact, evaluates only the frozen model and the baseline, and at most once per
     frozen configuration (ledger). After any test evaluation the validation stage refuses to run again.
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .data_io import read_jsonl, read_qrels
from .embed import CacheMismatch, cache_key, encode_texts, load_embeddings, save_embeddings
from .evaluate import METRIC_DEFINITIONS, evaluate_run, qrels_to_dict
from .index_faiss import IndexAlignmentError, build_index, check_unit_norm, load_index, save_index, search
from .train import RssMonitor
from .utils import get_logger, read_json, sha256_file, utc_now, write_json

log = get_logger("compare")


@dataclass
class Loaded:
    encode_fn: Callable[[list[str]], np.ndarray]     # raw (un-normalized) batch encoder
    prov: dict
    release: Callable[[], None] = lambda: None


class ProtocolError(Exception):
    """A pre-registered protocol rule would be violated (e.g. a second test evaluation)."""


# ------------------------------------------------------------------------------------------ statistics
def bootstrap_mean_ci(values, n_resamples: int, seed: int, confidence: float) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    idx = np.random.default_rng(seed).integers(0, len(v), size=(n_resamples, len(v)))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - confidence) / 2 * 100, (1 + confidence) / 2 * 100])
    return {"mean": float(v.mean()), "ci_low": float(lo), "ci_high": float(hi)}


def paired_bootstrap(a: dict[str, float], b: dict[str, float], n_resamples: int, seed: int, confidence: float) -> dict[str, Any]:
    """Per-query paired difference A - B with a percentile bootstrap CI (queries resampled together)."""
    if set(a) != set(b):
        raise ValueError("paired comparison requires identical query sets")
    qids = sorted(a)
    d = np.array([a[q] - b[q] for q in qids], dtype=np.float64)
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n_resamples, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - confidence) / 2 * 100, (1 + confidence) / 2 * 100])
    return {"n_queries": len(d), "mean_diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "share_resamples_positive": float((means > 0).mean()),
            "queries_better": int((d > 1e-12).sum()), "queries_tied": int((np.abs(d) <= 1e-12).sum()),
            "queries_worse": int((d < -1e-12).sum()),
            "note": "share_resamples_positive is a bootstrap share, NOT a p-value."}


def select_final(scores: dict[str, float], tie_break_order: list[str], tol: float = 1e-12) -> str:
    missing = set(scores) - set(tie_break_order)
    if missing:
        raise ValueError(f"tie_break_order lacks candidates {sorted(missing)}")
    best = max(scores.values())
    return next(k for k in tie_break_order if k in scores and abs(scores[k] - best) <= tol)


def fingerprint(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def frozen_inputs(cfg: dict, selected: str, weight_hashes: dict[str, str], corpus_sha: str, val_qrels_sha: str) -> dict:
    return {"selected": selected, "weights_sha256": weight_hashes, "corpus_sha256": corpus_sha, "val_qrels_sha256": val_qrels_sha,
            "embedding": {k: cfg["embedding"][k] for k in ("batch_size", "normalize", "dtype")},
            "max_seq_length": cfg["model"]["max_seq_length"], "top_k": cfg["retrieval"]["top_k"],
            "evaluation": cfg["evaluation"], "selection": cfg["selection"], "bootstrap": cfg["bootstrap"],
            "seed": cfg["project"]["seed"], "index": cfg["index"]["type"]}


# ------------------------------------------------------------------------------------------ test ledger
def ledger_entries(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []


def ledger_key(key: str, weights_sha: str, frozen_fp: str) -> str:
    return f"{key}:{weights_sha}:{frozen_fp}"


def ledger_assert_unused(path: Path, entry_keys: list[str]) -> None:
    used = {e["entry_key"] for e in ledger_entries(path)}
    clash = [k for k in entry_keys if k in used]
    if clash:
        raise ProtocolError(f"Held-out test already evaluated for {clash} (ledger {path.name}). One test evaluation per frozen "
                            "configuration is allowed; re-running would be test-guided iteration.")


def ledger_append(path: Path, entries: list[dict]) -> None:
    write_json_list(ledger_entries(path) + entries, path)


def write_json_list(obj: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------------------------ retrieval assets
def index_matches_embeddings(index, emb: np.ndarray) -> bool:
    return index.ntotal == len(emb) and bool(np.array_equal(index.reconstruct_n(0, index.ntotal), emb))


def build_assets(key: str, loaded: Loaded, doc_texts: list[str], doc_ids: list[str], corpus_hash: str, cfg: dict) -> dict:
    ec, tol = cfg["embedding"], cfg["checks"]["norm_tolerance"]
    emb_dir, idx_dir = cfg["_abs"]["embeddings_dir"], cfg["_abs"]["indexes_dir"]
    ck, stem, idx_stem = cache_key(corpus_hash, loaded.prov, ec["normalize"], ec["batch_size"]), f"{key}_docs", f"{key}_flatip"
    t0, used = time.perf_counter(), True
    try:
        emb = load_embeddings(emb_dir, stem, ck)
    except (FileNotFoundError, CacheMismatch) as exc:
        used = False
        log.info("[%s] encoding corpus (%s)", key, exc)
        emb = encode_texts(loaded.encode_fn, doc_texts, ec["chunk_size"], ec["normalize"])
        save_embeddings(emb, ck, {"provenance": loaded.prov}, emb_dir, stem)
    enc_s = 0.0 if used else time.perf_counter() - t0
    if emb.shape != (len(doc_ids), int(loaded.prov["embedding_dim"])) or emb.dtype != np.float32:
        raise ValueError(f"[{key}] unexpected embedding matrix {emb.shape} {emb.dtype}")
    norm_dev = check_unit_norm(emb, tol)
    index = build_index(emb, tol)
    idx_path, written = idx_dir / f"{idx_stem}.faiss", False
    if idx_path.exists():                                   # never overwrite an existing index with different content
        old, _ = load_index(idx_dir, idx_stem, expected_doc_ids=doc_ids)
        if not index_matches_embeddings(old, emb):
            raise IndexAlignmentError(f"Existing index {idx_path.name} does not match the cached embeddings; refusing to overwrite. "
                                      "Review it and move/delete it yourself.")
    else:
        save_index(index, doc_ids, idx_dir, idx_stem, {"model": loaded.prov, "corpus_sha256": corpus_hash,
                                                       "index_type": "IndexFlatIP", "normalized": True})
        written = True
    return {"emb": emb, "index": index, "cache_used": used, "encode_s": enc_s, "norm_dev": norm_dev, "idx_stem": idx_stem,
            "index_written": written, "index_sha256": sha256_file(idx_path), "emb_sha256": sha256_file(emb_dir / f"{stem}.npy")}


def evaluate_candidate(key: str, loaded: Loaded, assets: dict, doc_ids: list[str], doc_texts: list[str], q_ids: list[str],
                       q_texts: list[str], qrels: dict, cfg: dict, measure_latency: bool) -> dict:
    from sklearn.metrics.pairwise import cosine_similarity

    ec, top_k, ev, ck = cfg["embedding"], cfg["retrieval"]["top_k"], cfg["evaluation"], cfg["checks"]
    kw = dict(ks=tuple(ev["ks"]), ndcg_k=ev["ndcg_k"], mrr_k=ev["mrr_k"], precision_k=ev["precision_k"], map_k=ev["map_k"])
    with RssMonitor(1.0) as mon:
        t0 = time.perf_counter()
        q_emb = encode_texts(loaded.encode_fn, q_texts, ec["chunk_size"], ec["normalize"])
        q_enc_s = time.perf_counter() - t0
        index2, ids2 = load_index(cfg["_abs"]["indexes_dir"], assets["idx_stem"], expected_doc_ids=doc_ids)
        results = search(index2, ids2, q_emb, top_k)
        run = {q: [d for d, _ in r] for q, r in zip(q_ids, results)}
        res = evaluate_run(run, qrels, **kw)
        rerun = evaluate_run({q: [d for d, _ in r] for q, r in zip(q_ids, search(index2, ids2, q_emb, top_k))}, qrels, **kw)
        rng = np.random.default_rng(cfg["project"]["seed"])
        emb, n = assets["emb"], min(ck["n_consistency_samples"], len(q_ids))
        ip_np = ip_cos = 0.0
        for i in rng.choice(len(q_ids), size=n, replace=False):
            ref = cosine_similarity(q_emb[i:i + 1], emb)[0]
            for doc, score in results[i][:10]:
                j = ids2.index(doc)
                ip_np, ip_cos = max(ip_np, abs(score - float(q_emb[i] @ emb[j]))), max(ip_cos, abs(score - float(ref[j])))
        pick = rng.choice(len(doc_ids), size=min(ck["reproducibility_resample_docs"], len(doc_ids)), replace=False)
        repro = float(np.max(np.abs(encode_texts(loaded.encode_fn, [doc_texts[i] for i in pick], ec["chunk_size"], True) - emb[pick])))
        latency = None
        if measure_latency:
            enc_ms, srch_ms = [], []
            for k_, q in enumerate(q_texts):
                t = time.perf_counter(); e = np.asarray(loaded.encode_fn([q]), dtype=np.float32); t1 = time.perf_counter()
                e = e / max(float(np.linalg.norm(e)), 1e-12)
                search(index2, ids2, e.astype(np.float32), top_k); t2 = time.perf_counter()
                if k_ >= cfg["latency"]["n_warmup"]:
                    enc_ms.append((t1 - t) * 1e3); srch_ms.append((t2 - t1) * 1e3)
            pct = lambda a, p: float(np.percentile(a, p))
            latency = {"n_timed_queries": len(enc_ms), "encode_ms": {"mean": float(np.mean(enc_ms)), "p50": pct(enc_ms, 50), "p95": pct(enc_ms, 95)},
                       "faiss_search_ms": {"mean": float(np.mean(srch_ms)), "p50": pct(srch_ms, 50), "p95": pct(srch_ms, 95)}}
    checks = {"doc_index_mapping_consistent": ids2 == doc_ids and index2.ntotal == len(doc_ids),
              "reloaded_index_equals_in_memory_results": search(assets["index"], doc_ids, q_emb, top_k) == results,
              "scores_sorted_descending": all([s for _, s in r] == sorted((s for _, s in r), reverse=True) for r in results),
              "ip_equals_cosine": ip_np <= ck["score_tolerance"] and ip_cos <= ck["score_tolerance"],
              "reencode_reproduces_cache": repro <= ck["reproducibility_tolerance"],
              "metrics_reproduced_on_rerun": res["aggregate"] == rerun["aggregate"],
              "all_queries_evaluated": res["n_queries"] == len(q_ids) and not res["queries_missing_from_run"]}
    return {"scored": {q: [(d, s) for d, s in r] for q, r in zip(q_ids, results)}, "eval": res, "checks": checks,
            "numbers": {"ip_vs_numpy_max_abs_diff": ip_np, "ip_vs_sklearn_cosine_max_abs_diff": ip_cos, "reencode_max_abs_diff": repro,
                        "unit_norm_max_deviation": assets["norm_dev"]},
            "latency": latency, "timing_s": {"encode_corpus": assets["encode_s"], "encode_queries_batch": q_enc_s},
            "peak_rss_mb": round(mon.peak_mb, 1)}


# ------------------------------------------------------------------------------------------ reporting helpers
def best_positive_rank(ranked: list[str], positives: set[str]) -> int | None:
    return next((i + 1 for i, d in enumerate(ranked) if d in positives), None)


def rank_buckets(ranks: list[int | None], top_k: int) -> dict[str, int]:
    b = {"rank 1": 0, "rank 2-10": 0, "rank 11-%d" % top_k: 0, "not in top %d" % top_k: 0}
    for r in ranks:
        b["not in top %d" % top_k if r is None else ("rank 1" if r == 1 else "rank 2-10" if r <= 10 else "rank 11-%d" % top_k)] += 1
    return b


def write_per_query_csv(path: Path, per_query: dict[str, dict[str, float]]) -> None:
    cols = list(next(iter(per_query.values())))
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["query_id", *cols])
        for q in sorted(per_query):
            w.writerow([q, *[f"{per_query[q][c]:.6f}" for c in cols]])


def append_runs_csv(path: Path, rows: list[dict]) -> None:
    """Append to experiments/runs.csv using its existing header (Phase 2 format); unknown columns are left blank."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        header = next(csv.reader(path.open(encoding="utf-8", newline="")))
    else:
        header = ["timestamp_utc", "phase", "run_id", "model", "revision", "split", "n_queries", *rows[0]["_metric_cols"], "seed", "status"]
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(header)
        for r in rows:
            w.writerow([r.get(h, "") for h in header])


def failure_cases_markdown(title: str, sel: str, base: str, pq: dict, scored: dict, positives: dict, titles: dict, queries: dict,
                           n_cases: int, chars: int, top_k: int, note: str) -> str:
    pos_rank = {k: {q: best_positive_rank([d for d, _ in scored[k][q]], positives[q]) for q in positives} for k in (sel, base)}
    L = [f"# {title}", "", f"> {note}", "", "> Retrieved documents that are not judged positives may still be relevant (qrels are sparse).", ""]
    for k in dict.fromkeys((sel, base)):
        L += [f"**Best-positive rank distribution — {k}:** {rank_buckets(list(pos_rank[k].values()), top_k)}", ""]

    def block(q: str) -> list[str]:
        out = [f"### Query {q}", f"**Claim:** {queries[q]}", "",
               "**Judged positive(s):** " + "; ".join(f"[{d}] {titles[d][:chars]}" for d in sorted(positives[q])),
               f"- best-positive rank: {sel}={pos_rank[sel][q]}, {base}={pos_rank[base][q]}  |  nDCG@10: {sel}={pq[sel][q]['ndcg@10']:.3f}, {base}={pq[base][q]['ndcg@10']:.3f}"]
        out += [f"- {sel} #{i + 1} (cos {s:.3f}): [{d}] {titles[d][:chars]}" for i, (d, s) in enumerate(scored[sel][q][:3])]
        return out + [""]

    miss = sorted((q for q in positives if pq[sel][q]["recall@10"] == 0), key=lambda q: (pq[sel][q]["ndcg@10"], q))
    L += [f"## No judged positive in the top 10 ({len(miss)} of {len(positives)} queries; showing {min(n_cases, len(miss))})", ""]
    for q in miss[:n_cases]:
        L += block(q)
    if sel != base:
        delta = sorted(positives, key=lambda q: (pq[sel][q]["ndcg@10"] - pq[base][q]["ndcg@10"], q))
        L += [f"## Largest regressions vs {base} (nDCG@10)", ""]
        for q in [q for q in delta if pq[sel][q]["ndcg@10"] < pq[base][q]["ndcg@10"]][:n_cases]:
            L += block(q)
        L += [f"## Largest improvements vs {base} (nDCG@10)", ""]
        for q in [q for q in reversed(delta) if pq[sel][q]["ndcg@10"] > pq[base][q]["ndcg@10"]][:n_cases]:
            L += block(q)
    return "\n".join(L)


# ------------------------------------------------------------------------------------------ shared inputs
def _inputs(cfg: dict):
    proc = cfg["_abs"]["processed_dir"]
    docs = read_jsonl(proc / "corpus_processed.jsonl")
    doc_ids = [d["_id"] for d in docs]
    if doc_ids != read_json(proc / "document_ids.json"):
        raise ProtocolError("corpus_processed.jsonl order differs from document_ids.json; re-run prepare_data.py.")
    queries = {r["_id"]: r["text"] for r in read_jsonl(proc / "queries_processed.jsonl")}
    return docs, doc_ids, [d["doc_text"] for d in docs], {d["_id"]: d["title"] for d in docs}, queries, sha256_file(proc / "corpus_processed.jsonl")


def _phase4_gates(cfg: dict) -> dict[str, bool]:
    reports, root = cfg["_abs"]["reports_dir"], cfg["_root"]
    ver = reports / "finetune_verification.json"
    gates = {"phase4_verification_PASS": ver.is_file() and read_json(ver).get("status") == "PASS"}
    for key, c in cfg["candidates"].items():
        if c["kind"] != "local":
            continue
        rp = reports / f"finetune_{c['run_id']}_report.json"
        if not rp.is_file():
            gates[f"{key}:phase4_report_present"] = False
            continue
        rep = read_json(rp)
        w = next((p for p in (root / c["model_dir"] / "model.safetensors", root / c["model_dir"] / "0_Transformer" / "model.safetensors") if p.exists()), None)
        gates[f"{key}:weights_match_phase4_report"] = bool(w) and rep.get("model_weights_sha256") == sha256_file(w) and rep["status"] == "PASS"
    return gates


def _metric_cols(ev: dict) -> list[str]:
    return list(ev["aggregate"].keys())


# ------------------------------------------------------------------------------------------ VALIDATION stage
def run_validation_stage(cfg: dict, loader: Callable[[str, dict], Loaded], force: bool = False) -> dict:
    reports, root, proc = cfg["_abs"]["reports_dir"], cfg["_root"], cfg["_abs"]["processed_dir"]
    ledger_p, rep_p, sel_p = reports / "test_evaluation_ledger.json", reports / "model_comparison_val.json", reports / "model_selection.json"
    if ledger_entries(ledger_p):
        raise ProtocolError("A held-out test evaluation already exists. Re-running validation/selection now would be test-guided iteration.")
    if (rep_p.exists() or sel_p.exists()) and not force:
        raise SystemExit(f"{rep_p.name}/{sel_p.name} already exist. Re-run with --force to replace them (allowed only before any test evaluation).")
    docs, doc_ids, doc_texts, titles, queries, corpus_hash = _inputs(cfg)
    val_ids = read_json(proc / "val_query_ids.json")
    qrels = qrels_to_dict(read_qrels(proc / "val_qrels.tsv"))           # VALIDATION judgments only
    if set(qrels) != set(val_ids):
        raise ProtocolError("val_qrels.tsv queries differ from val_query_ids.json")
    q_ids = sorted(val_ids)
    q_texts = [queries[q] for q in q_ids]
    gates = _phase4_gates(cfg)
    top_k = cfg["retrieval"]["top_k"]
    out: dict[str, dict] = {}
    for key, cand in cfg["candidates"].items():
        log.info("=== candidate %s", key)
        loaded = loader(key, cand)
        try:
            assets = build_assets(key, loaded, doc_texts, doc_ids, corpus_hash, cfg)
            ec = evaluate_candidate(key, loaded, assets, doc_ids, doc_texts, q_ids, q_texts, qrels, cfg, measure_latency=True)
        finally:
            loaded.release()
        ec["assets"] = {k: assets[k] for k in ("cache_used", "index_written", "index_sha256", "emb_sha256")}
        ec["prov"] = loaded.prov
        ec["n_docs"] = assets["index"].ntotal
        out[key] = ec
        del assets
    # ---- per-model gates ----------------------------------------------------------------------------------
    for key, ec in out.items():
        gates.update({f"{key}:{k}": v for k, v in ec["checks"].items()})
    base_rep = reports / "baseline_val_metrics.json"
    if base_rep.is_file():
        ref = read_json(base_rep)["metrics"]
        diff = max(abs(ref[m] - out["baseline"]["eval"]["aggregate"][m]) for m in ref)
        gates["baseline:matches_phase2_report"] = diff <= cfg["gates"]["baseline_report_tol"]
    for key, cand in cfg["candidates"].items():
        rp = reports / f"finetune_{cand.get('run_id', '')}_report.json"
        if cand["kind"] == "local" and rp.is_file():
            v = read_json(rp)["validation"]
            gap = abs(v["best_primary"] - out[key]["eval"]["aggregate"]["ndcg@10"])
            out[key]["evaluator_agreement_abs_diff_ndcg10"] = gap
            gates[f"{key}:agrees_with_phase4_evaluator"] = gap <= cfg["gates"]["evaluator_agreement_tol"]
    gates["same_candidate_corpus_all_models"] = len({ec["n_docs"] for ec in out.values()}) == 1
    # ---- statistics, selection ------------------------------------------------------------------------------
    b = cfg["bootstrap"]
    pq = {k: ec["eval"]["per_query"] for k, ec in out.items()}
    summary = {k: {"metrics": ec["eval"]["aggregate"],
                   "ci": {m: bootstrap_mean_ci([pq[k][q][m] for q in q_ids], b["n_resamples"], b["seed"], b["confidence"]) for m in b["metrics"]}}
               for k, ec in out.items()}
    paired = {f"{a}__minus__{c}": {m: paired_bootstrap({q: pq[a][q][m] for q in q_ids}, {q: pq[c][q][m] for q in q_ids},
                                                       b["n_resamples"], b["seed"], b["confidence"]) for m in b["metrics"]}
              for a, c in cfg["comparisons"]}
    metric = cfg["selection"]["metric"]
    scores = {k: out[k]["eval"]["aggregate"][metric] for k in out}
    selected = select_final(scores, cfg["selection"]["tie_break_order"])
    # ---- artifacts --------------------------------------------------------------------------------------------
    reports.mkdir(parents=True, exist_ok=True)
    for k, ec in out.items():
        write_per_query_csv(reports / f"val_per_query_{k}.csv", ec["eval"]["per_query"])
        write_json({q: [[d, round(s, 6)] for d, s in r] for q, r in ec["scored"].items()}, reports / f"val_run_top100_{k}.json")
    positives = {q: set(qrels[q]) for q in q_ids}
    (reports / f"val_failure_cases_{selected}.md").write_text(failure_cases_markdown(
        f"Validation failure cases: {selected}", selected, "baseline", pq, {k: out[k]["scored"] for k in out}, positives, titles, queries,
        cfg["failure_cases"]["n_cases"], cfg["failure_cases"]["snippet_chars"], top_k,
        "Validation split: model-selection data (not the held-out test)."), encoding="utf-8")
    ok = all(v is True for v in gates.values())
    wh = {k: ec["prov"]["weights_sha256"] for k, ec in out.items()}
    inputs = frozen_inputs(cfg, selected, wh, corpus_hash, sha256_file(proc / "val_qrels.tsv"))
    fp = fingerprint(inputs)
    first = next(iter(out.values()))
    report = {"stage": "validation", "status": "PASS" if ok else "FAIL", "generated_utc": utc_now(), "seed": cfg["project"]["seed"],
              "eval_setup": {"split": "validation", "n_queries": len(q_ids), "n_judgments": first["eval"]["n_judgments"],
                             "candidate_corpus_size": first["n_docs"], "retrieval_top_k": top_k,
                             "same_metric_implementation": "src/evaluate.py", "metric_definitions": METRIC_DEFINITIONS,
                             "note": "Metrics compare against sparse known positives, not exhaustive relevance."},
              "models": {k: {"label": cfg["candidates"][k]["label"], "provenance": out[k]["prov"], "assets": out[k]["assets"],
                             "metrics": summary[k]["metrics"], "bootstrap_ci": summary[k]["ci"], "checks": out[k]["checks"],
                             "check_numbers": out[k]["numbers"], "latency_single_query": out[k]["latency"],
                             "timing_s": out[k]["timing_s"], "peak_rss_mb": out[k]["peak_rss_mb"],
                             "evaluator_agreement_abs_diff_ndcg10": out[k].get("evaluator_agreement_abs_diff_ndcg10"),
                             "best_positive_rank_buckets": rank_buckets([best_positive_rank([d for d, _ in out[k]["scored"][q]], positives[q]) for q in q_ids], top_k)}
                         for k in out},
              "paired_comparisons": paired, "selection": {**cfg["selection"], "scores": scores, "selected": selected},
              "gates": gates, "caveats": [
                  "Fine-tuned models' best epoch was chosen on these same validation queries inside Phase 4, so their validation scores are optimistic.",
                  "Selection uses the point estimate of validation nDCG@10; confidence intervals are reported but did not drive selection.",
                  "One training seed per model; seed-to-seed variance is not measured."]}
    write_json(report, rep_p)
    write_json({"status": "FROZEN" if ok else "NOT_FROZEN", "frozen_utc": utc_now(), "selected": selected, "selection_scores": scores,
                "rule": cfg["selection"], "fingerprint": fp, "fingerprint_inputs": inputs,
                "candidates": {k: {"weights_sha256": wh[k], "embeddings_sha256": out[k]["assets"]["emb_sha256"],
                                   "index_sha256": out[k]["assets"]["index_sha256"], "model_dir": cfg["candidates"][k].get("model_dir"),
                                   "validation_metrics": summary[k]["metrics"]} for k in out},
                "test_candidates": list(dict.fromkeys([selected, "baseline"])),
                "protocol": ["Selection was made on validation only, by the rule above, before test.tsv was read.",
                             "Only test_candidates are evaluated on test, once each (ledger).",
                             "Any change to weights, configs or inputs invalidates the fingerprint and blocks the test stage."]}, sel_p)
    rows = [{"timestamp_utc": utc_now(), "phase": 5, "run_id": f"{k}_val_shared", "model": out[k]["prov"]["model_name"],
             "revision": out[k]["prov"]["resolved_revision"], "split": "val", "n_queries": len(q_ids), "seed": cfg["project"]["seed"],
             "status": report["status"], "_metric_cols": _metric_cols(out[k]["eval"]), **out[k]["eval"]["aggregate"]} for k in out]
    append_runs_csv(root / "experiments" / "runs.csv", rows)
    return report


# ------------------------------------------------------------------------------------------ TEST stage
def run_test_stage(cfg: dict, loader: Callable[[str, dict], Loaded]) -> dict:
    reports, root, proc, raw = cfg["_abs"]["reports_dir"], cfg["_root"], cfg["_abs"]["processed_dir"], cfg["_raw"]
    sel_p, rep_p, ledger_p = reports / "model_selection.json", reports / "model_comparison_val.json", reports / "test_evaluation_ledger.json"
    if not sel_p.is_file() or not rep_p.is_file():
        raise ProtocolError("Run `python scripts/evaluate_models.py --stage validation` first: no freeze record / validation report.")
    sel, vrep = read_json(sel_p), read_json(rep_p)
    if sel.get("status") != "FROZEN" or vrep.get("status") != "PASS":
        raise ProtocolError(f"Selection is {sel.get('status')!r} / validation report {vrep.get('status')!r}; test evaluation requires FROZEN + PASS.")
    docs, doc_ids, doc_texts, titles, queries, corpus_hash = _inputs(cfg)
    current_wh = {k: sel["candidates"][k]["weights_sha256"] for k in sel["candidates"]}
    fp_now = fingerprint(frozen_inputs(cfg, sel["selected"], current_wh, corpus_hash, sha256_file(proc / "val_qrels.tsv")))
    if fp_now != sel["fingerprint"]:
        raise ProtocolError("Frozen configuration fingerprint changed (configs/data differ from the validation stage). Test evaluation blocked.")
    cand_keys = sel["test_candidates"]
    ledger_assert_unused(ledger_p, [ledger_key(k, sel["candidates"][k]["weights_sha256"], sel["fingerprint"]) for k in cand_keys])
    # ---- test data: raw hash must equal the Phase 1 audit; this is the ONLY place test.tsv is read ----------------------------
    audit = read_json(reports / "data_audit.json")
    if sha256_file(raw["test_qrels"]) != audit["raw_file_sha256"]["test_qrels"]:
        raise ProtocolError("data/raw/test.tsv does not match the SHA-256 recorded in the Phase 1 audit (immutable file changed).")
    test_ids = read_json(proc / "test_query_ids.json")
    qrels = qrels_to_dict(read_qrels(raw["test_qrels"]))
    train_ids, val_ids = read_json(proc / "train_query_ids.json"), read_json(proc / "val_query_ids.json")
    gates = {"selection_frozen_and_validation_PASS": True, "fingerprint_matches_freeze": True, "ledger_unused_for_these_models": True,
             "raw_test_file_matches_phase1_audit": True, "test_qrels_queries_equal_test_ids": set(qrels) == set(test_ids),
             "test_disjoint_from_train_and_val": not (set(test_ids) & (set(train_ids) | set(val_ids)))}
    q_ids = sorted(test_ids)
    q_texts = [queries[q] for q in q_ids]
    out: dict[str, dict] = {}
    for key in cand_keys:
        loaded = loader(key, cfg["candidates"][key])
        try:
            if loaded.prov["weights_sha256"] != sel["candidates"][key]["weights_sha256"]:
                raise ProtocolError(f"[{key}] weights differ from the frozen record; test evaluation blocked.")
            assets = build_assets(key, loaded, doc_texts, doc_ids, corpus_hash, cfg)
            if assets["emb_sha256"] != sel["candidates"][key]["embeddings_sha256"]:
                raise ProtocolError(f"[{key}] corpus embeddings differ from the frozen record.")
            ec = evaluate_candidate(key, loaded, assets, doc_ids, doc_texts, q_ids, q_texts, qrels, cfg, measure_latency=False)
        finally:
            loaded.release()
        ec["n_docs"] = assets["index"].ntotal
        ec["prov"] = loaded.prov
        out[key] = ec
        gates.update({f"{key}:{k}": v for k, v in ec["checks"].items()})
    gates["same_corpus_size_as_validation"] = all(ec["n_docs"] == vrep["eval_setup"]["candidate_corpus_size"] for ec in out.values())
    # ---- consume the single test evaluation IMMEDIATELY (metrics are stored in the ledger itself) --------------------------------
    ledger_append(ledger_p, [{"entry_key": ledger_key(k, sel["candidates"][k]["weights_sha256"], sel["fingerprint"]), "model": k,
                              "weights_sha256": sel["candidates"][k]["weights_sha256"], "frozen_fingerprint": sel["fingerprint"],
                              "evaluated_utc": utc_now(), "n_queries": len(q_ids), "metrics": out[k]["eval"]["aggregate"]} for k in out])
    # ---- reporting ---------------------------------------------------------------------------------------------------------
    b, top_k = cfg["bootstrap"], cfg["retrieval"]["top_k"]
    pq = {k: ec["eval"]["per_query"] for k, ec in out.items()}
    selected = sel["selected"]
    for k, ec in out.items():
        write_per_query_csv(reports / f"test_per_query_{k}.csv", ec["eval"]["per_query"])
        write_json({q: [[d, round(s, 6)] for d, s in r] for q, r in ec["scored"].items()}, reports / f"test_run_top100_{k}.json")
    positives = {q: set(qrels[q]) for q in q_ids}
    if len(out) == 2:
        (reports / f"test_failure_cases_{selected}.md").write_text(failure_cases_markdown(
            f"Held-out test failure cases: {selected}", selected, "baseline", pq, {k: out[k]["scored"] for k in out}, positives, titles, queries,
            cfg["failure_cases"]["n_cases"], cfg["failure_cases"]["snippet_chars"], top_k,
            "DESCRIPTIVE ONLY. The model is frozen; nothing may be tuned after reading these test cases."), encoding="utf-8")
    paired = None
    if selected != "baseline":
        paired = {m: paired_bootstrap({q: pq[selected][q][m] for q in q_ids}, {q: pq["baseline"][q][m] for q in q_ids},
                                      b["n_resamples"], b["seed"], b["confidence"]) for m in b["metrics"]}
    ok = all(v is True for v in gates.values())
    val_metrics = {k: vrep["models"][k]["metrics"] for k in out}
    report = {"stage": "test", "status": "PASS" if ok else "FAIL", "generated_utc": utc_now(), "seed": cfg["project"]["seed"],
              "selected_model": selected, "freeze_fingerprint": sel["fingerprint"],
              "eval_setup": {"split": "test (held-out)", "n_queries": len(q_ids), "n_judgments": next(iter(out.values()))["eval"]["n_judgments"],
                             "candidate_corpus_size": next(iter(out.values()))["n_docs"], "retrieval_top_k": top_k,
                             "same_metric_implementation": "src/evaluate.py", "metric_definitions": METRIC_DEFINITIONS,
                             "note": "Metrics compare against sparse known positives, not exhaustive relevance. Shared corpus across splits."},
              "models": {k: {"label": cfg["candidates"][k]["label"], "provenance": out[k]["prov"],
                             "weights_sha256": sel["candidates"][k]["weights_sha256"], "test_metrics": out[k]["eval"]["aggregate"],
                             "test_bootstrap_ci": {m: bootstrap_mean_ci([pq[k][q][m] for q in q_ids], b["n_resamples"], b["seed"], b["confidence"]) for m in b["metrics"]},
                             "validation_metrics": val_metrics[k], "validation_to_test_change": {m: out[k]["eval"]["aggregate"][m] - val_metrics[k][m] for m in val_metrics[k]},
                             "checks": out[k]["checks"], "check_numbers": out[k]["numbers"], "peak_rss_mb": out[k]["peak_rss_mb"],
                             "timing_s": out[k]["timing_s"],
                             "best_positive_rank_buckets": rank_buckets([best_positive_rank([d for d, _ in out[k]["scored"][q]], positives[q]) for q in q_ids], top_k)}
                         for k in out},
              "paired_selected_minus_baseline": paired, "gates": gates,
              "resource_usage_reference": {"see": "model_comparison_val.json models.*.latency_single_query / timing_s / peak_rss_mb (measured once, on validation queries)"},
              "caveats": ["One test evaluation per frozen model (ledger). No tuning or selection followed it.",
                          "Validation-to-test differences reflect different query sets and noise, not only generalization.",
                          "SciFact is a shared-corpus benchmark and qrels are sparse; metrics are relative to known positives."]}
    write_json(report, reports / "test_results.json")
    rows = [{"timestamp_utc": utc_now(), "phase": 5, "run_id": f"{k}_test_final", "model": out[k]["prov"]["model_name"],
             "revision": "", "split": "test", "n_queries": len(q_ids), "seed": cfg["project"]["seed"], "status": report["status"],
             "_metric_cols": _metric_cols(out[k]["eval"]), **out[k]["eval"]["aggregate"]} for k in out]
    append_runs_csv(root / "experiments" / "runs.csv", rows)
    return report
