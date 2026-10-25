"""Phase 5: shared-implementation comparison, validation-only model selection (frozen), and ONE held-out test evaluation.

Usage (PowerShell, repo root, env active):
    python scripts/evaluate_models.py --stage validation            # encodes corpus for the 2 fine-tuned models (baseline cache is reused)
    python scripts/evaluate_models.py --stage test --confirm-single-test-evaluation
The test stage reads data/raw/test.tsv, evaluates only the frozen model + baseline, and is blocked from repeating (ledger).
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compare import Loaded, ProtocolError, run_test_stage, run_validation_stage
from src.config import load_config

SHOW = ["ndcg@10", "recall@1", "recall@5", "recall@10", "mrr@10"]


def real_loader(cfg):
    def loader(key, cand):
        from src.embed import load_local_model, load_sentence_model, make_encode_fn
        mc = cfg["model"]
        if cand["kind"] == "pretrained":
            model, prov = load_sentence_model(mc["name"], mc["revision"], mc["max_seq_length"])
        else:
            model, prov = load_local_model(cfg["_root"] / cand["model_dir"], mc["max_seq_length"], cand["run_id"])
        holder = [model]
        return Loaded(make_encode_fn(model, cfg["embedding"]["batch_size"]), prov, lambda: (holder.clear(), gc.collect()))
    return loader


def table(models: dict, key: str, ci_key: str) -> None:
    print(f"{'model':<18}" + "".join(f"{m:>10}" for m in SHOW) + "   nDCG@10 95% CI")
    for k, v in models.items():
        m, ci = v[key], v[ci_key]["ndcg@10"]
        print(f"{k:<18}" + "".join(f"{m[x]:>10.4f}" for x in SHOW) + f"   [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["validation", "test"])
    ap.add_argument("--config", nargs="+", default=["configs/baseline.yaml", "configs/compare.yaml"])
    ap.add_argument("--force", action="store_true", help="validation only: replace existing comparison/selection (refused after any test evaluation)")
    ap.add_argument("--confirm-single-test-evaluation", action="store_true",
                    help="required for --stage test: acknowledges that this is the one held-out evaluation of the frozen model")
    a = ap.parse_args()
    cfg = load_config(a.config)
    try:
        if a.stage == "validation":
            rep = run_validation_stage(cfg, real_loader(cfg), force=a.force)
            print("\n=== VALIDATION (shared metric implementation, same corpus/queries/depth) ===")
            table(rep["models"], "metrics", "bootstrap_ci")
            for name, comp in rep["paired_comparisons"].items():
                c = comp["ndcg@10"]
                print(f"  {name:<42} nDCG@10 diff {c['mean_diff']:+.4f}  CI [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]  better/tied/worse {c['queries_better']}/{c['queries_tied']}/{c['queries_worse']}")
            print(f"\nSELECTED (pre-registered rule, validation only): {rep['selection']['selected']}  scores={ {k: round(v, 4) for k, v in rep['selection']['scores'].items()} }")
            print("status:", rep["status"], "| failed gates:", [k for k, v in rep["gates"].items() if v is not True] or "none")
            return 0 if rep["status"] == "PASS" else 1
        if not a.confirm_single_test_evaluation:
            print("Refusing: the test stage is a one-shot evaluation. Re-run with --confirm-single-test-evaluation.")
            return 2
        rep = run_test_stage(cfg, real_loader(cfg))
        print("\n=== HELD-OUT TEST (frozen model + baseline; one evaluation) ===")
        table({k: {"metrics": v["test_metrics"], "ci": v["test_bootstrap_ci"]} for k, v in rep["models"].items()}, "metrics", "ci")
        if rep["paired_selected_minus_baseline"]:
            c = rep["paired_selected_minus_baseline"]["ndcg@10"]
            print(f"  {rep['selected_model']} minus baseline: nDCG@10 {c['mean_diff']:+.4f}  CI [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]  better/tied/worse {c['queries_better']}/{c['queries_tied']}/{c['queries_worse']}")
        print("status:", rep["status"], "| failed gates:", [k for k, v in rep["gates"].items() if v is not True] or "none")
        return 0 if rep["status"] == "PASS" else 1
    except ProtocolError as exc:
        print(f"PROTOCOL BLOCK: {exc}")
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
