"""Phase 6 exit gates for a built package: fresh reload reproduces Phase 5 rankings (ties aside), results are well-formed,
invalid input is handled, scores are true cosines, and latency is benchmarked AFTER startup (startup reported separately)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
import psutil

from .data_io import read_qrels
from .data_io import read_jsonl
from .embed import l2_normalize
from .evaluate import evaluate_run, qrels_to_dict
from .search_service import SearchService, benchmark_service, compare_rankings, verify_package
from .train import RssMonitor
from .utils import read_json, utc_now, write_json


def _behaviour(svc: SearchService) -> tuple[dict[str, bool], dict[str, str]]:
    """Exercise invalid / unusual input. Returns (gate booleans, human-readable observed behaviour)."""
    obs, ok = {}, {}

    def raises(exc, *a, **k):
        try:
            svc.search(*a, **k)
        except exc:
            return True
        except Exception:  # noqa: BLE001
            return False
        return False

    r_empty, r_ws = svc.search_detailed("", 5), svc.search_detailed("  \n\t ", 5)
    ok["empty_and_whitespace_query_return_no_hits_gracefully"] = (r_empty.status == r_ws.status == "empty_query" and not r_empty.hits and not r_ws.hits)
    obs["empty / whitespace query"] = "status=empty_query, 0 hits (no exception)"
    ok["non_string_query_raises_TypeError"] = raises(TypeError, None) and raises(TypeError, 123) and raises(TypeError, ["a"])
    ok["invalid_top_k_raises"] = (raises(ValueError, "x y", 0) and raises(ValueError, "x y", -3) and raises(TypeError, "x y", 1.5)
                                  and raises(TypeError, "x y", True) and raises(TypeError, "x y", "5"))
    obs["top_k 0 / negative"] = "ValueError"; obs["top_k float / bool / str"] = "TypeError"
    big = svc.search_detailed("protein", 10**6)
    ok["huge_top_k_is_clipped_and_flagged"] = big.top_k_effective == min(svc.max_top_k, svc._index.ntotal) and len(big.hits) == big.top_k_effective and bool(big.warnings)
    obs["top_k = 1,000,000"] = f"clipped to {big.top_k_effective} with warning"
    ok["default_top_k_used_when_none"] = len(svc.search("protein")) == min(svc.default_top_k, svc._index.ntotal)
    long_r = svc.search_detailed("protein " * 50000, 3)
    ok["very_long_query_truncated_not_crashing"] = long_r.status == "ok" and any("truncated" in w for w in long_r.warnings) and len(long_r.hits) == min(3, svc._index.ntotal)
    uni = svc.search("β-catenin αβ 研究 café", 3)
    ok["unicode_query_ok"] = len(uni) == min(3, svc._index.ntotal)
    a, b = svc.search("wnt  signalling\n in\tcells", 5), svc.search("wnt signalling in cells", 5)
    ok["whitespace_variants_give_identical_results"] = [h.corpus_id for h in a] == [h.corpus_id for h in b]
    ok["nul_character_handled"] = len(svc.search("wnt\x00signalling", 2)) >= 1
    return ok, obs


def run_service_validation(cfg: dict, package_dir: Path, encoder_factory: Callable | None = None) -> dict:
    reports, proc = cfg["_abs"]["reports_dir"], cfg["_abs"]["processed_dir"]
    vc, bc = cfg["validation"], cfg["benchmark"]
    t_all = time.perf_counter()
    ver = verify_package(package_dir)
    svc1 = SearchService.load(package_dir, verify=True, encoder_factory=encoder_factory)
    rss_after_load = psutil.Process().memory_info().rss / 1e6
    svc2 = SearchService.load(package_dir, verify=False, encoder_factory=encoder_factory)    # independent second load
    key = svc1.manifest["selected_model"]

    ref = read_json(reports / f"val_run_top100_{key}.json")
    val_ids = sorted(read_json(proc / "val_query_ids.json"))
    queries = {r["_id"]: r["text"] for r in read_jsonl(proc / "queries_processed.jsonl")}
    qrels = qrels_to_dict(read_qrels(proc / "val_qrels.tsv"))                              # validation judgments only
    depth = len(next(iter(ref.values())))
    if depth > svc1.max_top_k:
        raise ValueError(f"reference depth {depth} exceeds service max_top_k {svc1.max_top_k}")

    exact = tie = failed = 0
    failures, run1, responses, identical_loads = [], {}, [], True
    for q in val_ids:
        r1 = svc1.search_detailed(queries[q], depth)
        r2 = svc2.search_detailed(queries[q], depth)
        responses.append(r1)
        got = [(h.corpus_id, h.similarity_score) for h in r1.hits]
        run1[q] = [d for d, _ in got]
        cmp = compare_rankings([(d, s) for d, s in ref[q]], got, vc["tie_tolerance"])
        exact += cmp["exact"]; tie += (cmp["equivalent"] and not cmp["exact"]); failed += (not cmp["equivalent"])
        if not cmp["equivalent"]:
            failures.append(q)
        identical_loads &= compare_rankings(got, [(h.corpus_id, h.similarity_score) for h in r2.hits], 1e-9)["equivalent"]

    res = evaluate_run(run1, qrels, ks=tuple(cfg["evaluation"]["ks"]), ndcg_k=cfg["evaluation"]["ndcg_k"], mrr_k=cfg["evaluation"]["mrr_k"],
                       precision_k=cfg["evaluation"]["precision_k"], map_k=cfg["evaluation"]["map_k"])
    ref_ndcg = read_json(reports / "model_comparison_val.json")["models"][key]["metrics"]["ndcg@10"]
    ndcg_diff = abs(res["aggregate"]["ndcg@10"] - ref_ndcg)

    # ---- cosine / index / schema checks -------------------------------------------------------------------------------
    emb = np.load(package_dir / "embeddings" / "docs.npy")
    ids = svc1._ids
    rng = np.random.default_rng(cfg["project"]["seed"])
    sc_diff = 0.0
    for i in rng.choice(len(val_ids), size=min(vc["n_score_samples"], len(val_ids)), replace=False):
        qe = l2_normalize(np.asarray(svc1._encode([queries[val_ids[i]]]), dtype=np.float32))[0]
        for h in responses[i].hits[:10]:
            sc_diff = max(sc_diff, abs(h.similarity_score - float(qe @ emb[ids.index(h.corpus_id)])))
    idx_equals_emb = bool(np.array_equal(svc1._index.reconstruct_n(0, svc1._index.ntotal), emb))
    idset = set(ids)
    schema_ok = all(
        [h.rank for h in r.hits] == list(range(1, len(r.hits) + 1)) and len({h.corpus_id for h in r.hits}) == len(r.hits)
        and all(h.corpus_id in idset and h.title and isinstance(h.similarity_score, float) and len(h.snippet) <= svc1.snippet_chars + 2 for h in r.hits)
        and all(a.similarity_score >= b.similarity_score for a, b in zip(r.hits, r.hits[1:])) for r in responses)

    behaviour_ok, behaviour_obs = _behaviour(svc1)
    examples = {}
    for q in vc["example_queries"]:
        r = svc1.search_detailed(q, 3)
        examples[q] = [{"rank": h.rank, "corpus_id": h.corpus_id, "title": h.title, "similarity_score": round(h.similarity_score, 4)} for h in r.hits]

    # ---- benchmark (after startup) ---------------------------------------------------------------------------------------
    with RssMonitor(0.5) as mon:
        bench = benchmark_service(svc1, [queries[q] for q in val_ids], [k for k in bc["top_ks"] if k <= svc1.max_top_k], bc["n_warmup"], bc["repeats"])
    try:
        import torch
        threads = torch.get_num_threads()
    except Exception:  # noqa: BLE001
        threads = None

    gates = {"package_hashes_verified": ver["ok"],
             "reloaded_rankings_match_phase5_run_tie_aware": failed == 0 and len(val_ids) == len(ref),
             "two_independent_loads_identical": bool(identical_loads),
             "service_val_ndcg10_matches_phase5": ndcg_diff <= vc["ndcg_tolerance"],
             "scores_are_true_cosines": sc_diff <= vc["score_tolerance"],
             "faiss_index_equals_packaged_embeddings": idx_equals_emb,
             "result_schema_ordering_uniqueness": schema_ok,
             **{f"input:{k}": v for k, v in behaviour_ok.items()}}
    report = {"stage": "service_validation", "status": "PASS" if all(gates.values()) else "FAIL", "generated_utc": utc_now(),
              "package_dir": package_dir.name, "model": key, "freeze_fingerprint": svc1.manifest["freeze_fingerprint"], "gates": gates,
              "reload_consistency": {"n_queries": len(val_ids), "compared_depth": depth, "exact": exact, "tie_aware_equivalent_not_exact": tie,
                                     "not_equivalent": failed, "failing_queries": failures[:10], "tie_tolerance": vc["tie_tolerance"]},
              "end_to_end_metric": {"service_val_ndcg@10": res["aggregate"]["ndcg@10"], "phase5_val_ndcg@10": ref_ndcg, "abs_diff": ndcg_diff,
                                    "note": "validation split only; test.tsv not used in this phase"},
              "score_check_max_abs_diff": sc_diff, "input_handling_observed": behaviour_obs, "example_queries": examples,
              "startup_s_excluded_from_latency": svc1.startup_s, "latency_after_startup": bench,
              "memory_mb": {"rss_after_load": round(rss_after_load, 1), "peak_rss_during_benchmark": round(mon.peak_mb, 1),
                            "note": "RSS of the whole Python process (two services were loaded for the reload check)"},
              "package": {"n_files": ver["n_files"], "size_mb": round(sum(m["size"] for m in svc1.manifest["files"].values()) / 1e6, 1),
                          "extra_files": ver["extra"]},
              "environment": {"torch_threads": threads, "cpu_count": psutil.cpu_count(logical=True)}, "elapsed_s": round(time.perf_counter() - t_all, 1),
              "caveats": ["Latency is single-user, sequential, CPU-only on this machine; not a throughput guarantee.",
                          "similarity_score is cosine similarity, not a probability of relevance or factual correctness."]}
    write_json(report, reports / "search_service_report.json")
    return report
