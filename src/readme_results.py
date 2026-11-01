"""Fill the generated blocks of README.md / docs/MODEL_CARD.md from the measured report files.
Hand-written prose never contains results: numbers appear ONLY between <!-- NAME:START --> and <!-- NAME:END --> markers and are produced here.
Missing reports yield an explicit 'not generated yet' line, never a guess."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPORTS = {"audit": "data_audit.json", "val": "model_comparison_val.json", "sel": "model_selection.json", "test": "test_results.json",
           "service": "search_service_report.json", "cal": "rag_evidence_calibration.json", "rag": "rag_validation_report.json",
           "app": "app_smoke_report.json"}
CMD = {"val": "python scripts/evaluate_models.py --stage validation", "test": "python scripts/evaluate_models.py --stage test --confirm-single-test-evaluation",
       "service": "python scripts/validate_search_service.py", "cal": "python scripts/calibrate_rag_gate.py", "rag": "python scripts/validate_rag.py",
       "audit": "python scripts/prepare_data.py", "app": "python scripts/smoke_test_app.py --serve-check"}
SHOW = [("ndcg@10", "nDCG@10"), ("recall@1", "Recall@1"), ("recall@5", "Recall@5"), ("recall@10", "Recall@10"), ("mrr@10", "MRR@10")]


def load_reports(reports_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, name in REPORTS.items():
        p = Path(reports_dir) / name
        out[k] = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
    out["finetune"] = {}
    for p in sorted(Path(reports_dir).glob("finetune_*_report.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if "run_id" in d and not d.get("smoke"):
            out["finetune"][d["run_id"]] = d
    out["review"] = {p.stem.replace("rag_review_summary_", ""): json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(reports_dir).glob("rag_review_summary_*.json"))}
    return out


def _f(x: float | None, nd: int = 4) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def _ci(c: dict) -> str:
    return f"[{c['ci_low']:.4f}, {c['ci_high']:.4f}]"


def _missing(key: str) -> str:
    return f"_Not generated yet: run `{CMD[key]}`._\n"


def _table(header: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|", *["| " + " | ".join(r) + " |" for r in rows]]) + "\n"


def headline(R: dict) -> str:
    t, sel = R["test"], R["sel"]
    if not t or not sel:
        return "_Headline results are generated from the measured reports; run the pipeline, then `python scripts/generate_readme_results.py`._\n"
    s, m = t["selected_model"], t["models"]
    n = t["eval_setup"]["n_queries"]
    if s == "baseline":
        b = m["baseline"]
        return (f"**On {n} held-out SciFact test queries the pretrained baseline reached nDCG@10 {_f(b['test_metrics']['ndcg@10'])} {_ci(b['test_bootstrap_ci']['ndcg@10'])}; "
                f"fine-tuning did not beat it on validation, so the baseline was the frozen model.**\n")
    a, b, d = m[s], m["baseline"], (t.get("paired_selected_minus_baseline") or {}).get("ndcg@10")
    paired = f" The paired difference is {d['mean_diff']:+.4f} (95% CI [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}])." if d else ""
    return (f"**On {n} held-out SciFact test queries, the fine-tuned retriever reaches nDCG@10 {_f(a['test_metrics']['ndcg@10'])} {_ci(a['test_bootstrap_ci']['ndcg@10'])} "
            f"versus {_f(b['test_metrics']['ndcg@10'])} {_ci(b['test_bootstrap_ci']['ndcg@10'])} for the pretrained `all-MiniLM-L6-v2` baseline.{paired}** "
            f"One training seed; sparse judgments (see Limitations).\n")


def _metric_table(models: dict, key_metrics: str, key_ci: str, label_of) -> str:
    rows = [[label_of(k), *[_f(v[key_metrics][m]) for m, _ in SHOW], _ci(v[key_ci]["ndcg@10"])] for k, v in models.items()]
    return _table(["Model", *[n for _, n in SHOW], "nDCG@10 95% CI"], rows)


def results(R: dict) -> str:
    out = []
    a = R["audit"]
    out.append("### Data\n")
    if a and "reconciliation" in a and "corpus" in a:
        rc = a["reconciliation"]
        out.append(f"{a['corpus']['n_documents']:,} abstracts; {rc['queries_jsonl_total']:,} claims (queries): {rc['split_train_queries']} train / {rc['split_val_queries']} validation "
                   f"(seeded, query-grouped split of the official train queries) / {rc['official_test_queries']} held-out test.\n")
    else:
        out.append(_missing("audit"))
    v = R["val"]
    out.append("### Validation (model selection)\n")
    if v:
        es = v["eval_setup"]
        out.append(f"{es['n_queries']} queries, {es['n_judgments']} judgments, {es['candidate_corpus_size']:,}-document candidate corpus, top-{es['retrieval_top_k']} retrieval, one shared metric implementation for every model.\n")
        out.append(_metric_table(v["models"], "metrics", "bootstrap_ci", lambda k: v["models"][k]["label"]))
        rows = []
        for name, comp in v["paired_comparisons"].items():
            c = comp["ndcg@10"]
            rows.append([name.replace("__minus__", " − "), f"{c['mean_diff']:+.4f}", f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]", "yes" if c["ci_excludes_zero"] else "no", f"{c['queries_better']}/{c['queries_tied']}/{c['queries_worse']}"])
        out.append("\nPaired per-query nDCG@10 differences (bootstrap):\n")
        out.append(_table(["Comparison (A − B)", "Mean diff", "95% CI", "CI excludes 0", "Better/tied/worse"], rows))
        out.append(f"Selected on validation by the pre-registered rule: **{v['selection']['selected']}**. "
                   "The fine-tuned models' best epoch was also chosen on these queries, so their validation scores are optimistic.\n")
    else:
        out.append(_missing("val"))
    t = R["test"]
    out.append("### Held-out test (evaluated once)\n")
    if t:
        es = t["eval_setup"]
        out.append(f"{es['n_queries']} queries, {es['n_judgments']} judgments. The frozen model and the baseline were each evaluated exactly once (ledger-enforced); no tuning followed.\n")
        out.append(_metric_table({k: {"m": x["test_metrics"], "ci": x["test_bootstrap_ci"]} for k, x in t["models"].items()}, "m", "ci", lambda k: t["models"][k]["label"]))
        d = t.get("paired_selected_minus_baseline")
        if d:
            c = d["ndcg@10"]
            out.append(f"\nPaired nDCG@10 difference ({t['selected_model']} − baseline): **{c['mean_diff']:+.4f}**, 95% CI [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]; "
                       f"queries better/tied/worse: {c['queries_better']}/{c['queries_tied']}/{c['queries_worse']}.\n")
        sh = {k: x["validation_to_test_change"].get("ndcg@10") for k, x in t["models"].items()}
        out.append("\nChange in nDCG@10 from validation to test: " + ", ".join(f"{k} {x:+.4f}" for k, x in sh.items()) + ".\n")
    else:
        out.append(_missing("test"))
    out.append("### Training (CPU, 8 GB RAM class)\n")
    if R["finetune"]:
        rows = [[k, ", ".join(["pairs" if not r["uses_mined_negatives"] else "pairs + mined negatives"]), str(r["hyperparameters"]["epochs"]), str(r["validation"]["best_stage"]),
                 _f(r["validation"]["pre_training_primary"]), _f(r["validation"]["best_primary"]), f"{r['timing_s']['train_wall']:.0f}", f"{r['memory_mb']['peak_rss_observed']:.0f}"]
                for k, r in R["finetune"].items()]
        out.append(_table(["Run", "Training data", "Epochs", "Best stage", "Val nDCG@10 before", "Val nDCG@10 after", "Train time (s)", "Peak RSS (MB)"], rows))
        out.append("Validation numbers in this table come from the Sentence-Transformers evaluator; the comparison tables above use the project's own metric code (the two agreed on the pretrained model).\n")
    else:
        out.append("_No fine-tuning reports found._\n")
    s = R["service"]
    out.append("### Serving (packaged exact-FAISS service, CPU, sequential single-user)\n")
    if s:
        lat = s["latency_after_startup"]
        rows = [[k.replace("top_k=", "top-k "), _f(b["total_ms"]["p50"], 1), _f(b["total_ms"]["p95"], 1), _f(b["encode_ms_mean"], 1), _f(b["faiss_search_ms_mean"], 2), _f(b["throughput_qps_sequential"], 1)] for k, b in lat.items()]
        out.append(_table(["Setting", "p50 ms", "p95 ms", "Encode ms (mean)", "FAISS ms (mean)", "Queries/s"], rows))
        su = s["startup_s_excluded_from_latency"]
        out.append(f"Startup (excluded above): {su['total']:.1f} s total (model load {su['load_model']:.1f} s, hash verification {su['verify_hashes']:.1f} s). "
                   f"Package {s['package']['size_mb']} MB; process RSS after load {s['memory_mb']['rss_after_load']:.0f} MB. "
                   f"Reload reproduced Phase-5 rankings exactly for {s['reload_consistency']['exact']}/{s['reload_consistency']['n_queries']} validation queries.\n")
    else:
        out.append(_missing("service"))
    out.append("### Optional RAG layer\n")
    c, g = R["cal"], R["rag"]
    if c and g:
        cov = g["in_domain_validation_claims"]
        out.append(f"Evidence gate: top-1 cosine ≥ {c['recommended_min_top_score']} (rule: just above the highest off-topic score); "
                   f"{c['in_domain']['pass_rate_at_tau']:.1%} of {c['in_domain']['n']} validation claims pass; "
                   f"{g['off_topic']['refused']}/{g['off_topic']['n']} off-topic/gibberish queries refused without calling a generator. "
                   f"Guardrail battery: {g['battery']['n_results'] - g['battery']['n_failures']}/{g['battery']['n_results']} deliberately bad generated answers handled as required. "
                   f"Validation status: {g['status']}.\n")
        for mode, r in R["review"].items():
            a2 = r["answered_rows_labeled"]["answer_supported_yes"]
            out.append(f"Manual review ({mode}, {r['status']}): answers judged fully supported {a2['k']}/{a2['n']}"
                       + (f" (95% CI {a2['ci_low']:.0%}–{a2['ci_high']:.0%})" if a2["n"] else "") + ". Human-labelled sample; small and reviewer-specific.\n")
        if not R["review"]:
            out.append("Manual grounding review: not completed. Automatic checks verify structure, not factual correctness.\n")
    else:
        out.append(_missing("cal") if not c else _missing("rag"))
    sel = R["sel"]
    if sel:
        out.append("\n### Provenance\n")
        out.append(f"Freeze fingerprint `{sel['fingerprint'][:16]}…`; selected model weights SHA-256 `{sel['candidates'][sel['selected']]['weights_sha256'][:16]}…`; "
                   f"seed {sel['fingerprint_inputs']['seed']}. Full hashes: `results/reports/model_selection.json`.\n")
    return "\n".join(out)


def model_metrics(R: dict) -> str:
    t = R["test"]
    if not t:
        return _missing("test")
    s = t["selected_model"]
    rows = [[k, *[_f(x["test_metrics"][m]) for m, _ in SHOW]] for k, x in t["models"].items()]
    return f"Held-out test ({t['eval_setup']['n_queries']} queries); selected model: `{s}`.\n\n" + _table(["Model", *[n for _, n in SHOW]], rows)


def screenshots(root: Path) -> str:
    imgs = sorted(p for p in (Path(root) / "docs" / "screenshots").glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif"}) if (Path(root) / "docs" / "screenshots").is_dir() else []
    return "\n".join(f"![{p.stem.replace('_', ' ')}](docs/screenshots/{p.name})" for p in imgs) + ("\n" if imgs else "")


def replace_block(text: str, name: str, body: str) -> str:
    rx = re.compile(rf"(<!-- {name}:START -->\n)(.*?)(<!-- {name}:END -->)", re.S)
    if not rx.search(text):
        raise ValueError(f"markers <!-- {name}:START --> / <!-- {name}:END --> not found")
    return rx.sub(lambda m: m.group(1) + body.rstrip("\n") + "\n" + m.group(3), text, count=1)


def render_targets(R: dict, root: Path) -> dict[str, dict[str, str]]:
    return {"README.md": {"HEADLINE": headline(R), "RESULTS": results(R), "SCREENSHOTS": screenshots(root)},
            "docs/MODEL_CARD.md": {"MODEL_METRICS": model_metrics(R)}}
