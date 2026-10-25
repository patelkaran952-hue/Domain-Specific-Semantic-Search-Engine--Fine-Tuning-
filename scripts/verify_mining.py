"""Phase 3 exit gates. Re-reads mining outputs from disk and checks them independently of the mining run.
Status: PASS (all required sources present and clean) | PARTIAL (a required source has not been mined yet) | FAIL.
Usage: python scripts/verify_mining.py [--sources bm25 dense]
Only ID lists (never val/test qrels) are used for leakage checks."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data_io import read_jsonl, read_qrels
from src.negative_mining import read_jsonl_rows, verify_mining_outputs
from src.utils import read_json, sha256_file, utc_now, write_json

ALLOWED_DIFF = {"run_id", "data_file", "columns", "uses_mined_negatives", "negative_source", "caveat"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=None)
    a = ap.parse_args()
    cfg = load_config(["configs/baseline.yaml", "configs/mining.yaml"])
    root, mdir, proc, reports = cfg["_root"], cfg["_abs"]["mining_dir"], cfg["_abs"]["processed_dir"], cfg["_abs"]["reports_dir"]
    required = a.sources or cfg["mining"]["sources"]
    train_rows = read_qrels(proc / "train_qrels.tsv")
    ids = {n: read_json(proc / f"{n}_query_ids.json") for n in ("train", "val", "test")}
    corpus_ids = {d["_id"] for d in read_jsonl(proc / "corpus_processed.jsonl")}
    cur_hash = {"corpus_processed.jsonl": sha256_file(proc / "corpus_processed.jsonl"),
                "train_qrels.tsv": sha256_file(proc / "train_qrels.tsv"),
                "train_query_ids.json": sha256_file(proc / "train_query_ids.json")}

    per_source, missing = {}, []
    for s in required:
        rep_p, set_p = reports / f"negative_mining_{s}_report.json", mdir / f"mining_settings_{s}.json"
        if not rep_p.is_file() or not (mdir / f"hard_negatives_{s}.jsonl").is_file():
            missing.append(s)
            continue
        g = verify_mining_outputs(mdir, s, train_rows, ids["train"], ids["val"], ids["test"], corpus_ids, cfg["mining"]["pool_size"])
        settings = read_json(set_p)
        g["mining_report_status_PASS"] = read_json(rep_p)["status"] == "PASS"
        g["inputs_unchanged_since_mining"] = settings["inputs_sha256"] == cur_hash
        g["seed_matches_config"] = settings["seed"] == cfg["project"]["seed"]
        per_source[s] = g

    ft = yaml.safe_load((root / "configs" / "finetune.yaml").read_text(encoding="utf-8"))
    po, hn = ft["experiments"]["positive_only"], ft["experiments"]["hard_negative"]
    src = hn.get("negative_source")
    comparability = {
        "single_shared_training_block": isinstance(ft.get("training"), dict),
        "experiments_differ_only_in_data_and_columns": all(set(e) <= ALLOWED_DIFF for e in (po, hn)),
        "positive_only_has_no_negative_column": "negative" not in po["columns"] and not po["uses_mined_negatives"],
        "hard_negative_loss_consumes_negative_column": hn["columns"] == ["anchor", "positive", "negative"] and hn["uses_mined_negatives"],
        "positive_only_data_exists": (root / po["data_file"]).is_file(),
        "hard_negative_data_exists": True if (root / hn["data_file"]).is_file() else None,  # None = not mined yet (pending)
        "hard_negative_source_is_mined": src in cfg["mining"]["sources"],
        "same_train_pairs_count": None,
    }
    if comparability["positive_only_data_exists"] and comparability["hard_negative_data_exists"] is True:
        n_pairs, n_trip = len(read_jsonl_rows(root / po["data_file"])), len(read_jsonl_rows(root / hn["data_file"]))
        comparability["same_train_pairs_count"] = n_pairs == n_trip
        comparability["counts"] = {"pairs": n_pairs, "triplets": n_trip}
    else:
        comparability["same_train_pairs_count"] = None

    failed = {s: [k for k, v in g.items() if v is False] for s, g in per_source.items()}
    comp_failed = [k for k, v in comparability.items() if v is False]
    if any(failed.values()) or comp_failed:
        status = "FAIL"
    elif missing or comparability["same_train_pairs_count"] is None or comparability["hard_negative_data_exists"] is None:
        status = "PARTIAL"
    else:
        status = "PASS"
    out = {"status": status, "generated_utc": utc_now(), "required_sources": required, "missing_sources": missing,
           "per_source_gates": per_source, "failed_gates": failed, "finetune_comparability": comparability,
           "notes": ["Gates read only training qrels and ID lists; no validation/test judgments.",
                     "Hard negatives are unjudged; false-negative risk is flagged, not eliminated."]}
    write_json(out, reports / "negative_mining_verification.json")
    print(f"Phase 3 verification: {status} | missing sources: {missing} | failed: {failed} {comp_failed}")
    return {"PASS": 0, "PARTIAL": 3, "FAIL": 1}[status]


if __name__ == "__main__":
    raise SystemExit(main())
