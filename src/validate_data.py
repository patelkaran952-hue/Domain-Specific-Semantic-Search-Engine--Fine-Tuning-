"""Schema / ID / consistency validation and statistics for the SciFact files."""
from __future__ import annotations

import statistics
from collections import Counter
from typing import Any


class DataValidationError(Exception):
    """Hard validation failure (the pipeline must stop)."""


def _is_blank(x: Any) -> bool:
    return not isinstance(x, str) or not x.strip()


def validate_corpus(records: list[dict]) -> dict[str, Any]:
    """Hard errors: missing fields, non-string/blank/duplicate _id, non-string title/text.
    Warnings (reported, not fatal): blank title/text, metadata not a dict."""
    errors: list[str] = []
    seen: Counter = Counter()
    blank_title, blank_text, bad_meta = [], [], []
    for i, rec in enumerate(records):
        for field in ("_id", "title", "text"):
            if field not in rec:
                errors.append(f"corpus record #{i}: missing field {field!r}")
        did = rec.get("_id")
        if not isinstance(did, str) or not did.strip():
            errors.append(f"corpus record #{i}: _id must be a non-blank string, got {did!r}")
            continue
        seen[did] += 1
        if "title" in rec and not isinstance(rec["title"], str):
            errors.append(f"corpus {did}: title must be a string")
        if "text" in rec and not isinstance(rec["text"], str):
            errors.append(f"corpus {did}: text must be a string")
        if _is_blank(rec.get("title")):
            blank_title.append(did)
        if _is_blank(rec.get("text")):
            blank_text.append(did)
        if not isinstance(rec.get("metadata", {}), dict):
            bad_meta.append(did)
    dups = sorted(d for d, c in seen.items() if c > 1)
    if dups:
        errors.append(f"duplicate corpus _id values ({len(dups)}): {dups[:10]}")
    if errors:
        raise DataValidationError("Corpus validation failed:\n  " + "\n  ".join(errors[:20]))
    title_words = [len(r["title"].split()) for r in records]
    text_words = [len(r["text"].split()) for r in records]
    return {
        "n_documents": len(records),
        "n_unique_ids": len(seen),
        "blank_title_ids": blank_title,
        "blank_text_ids": blank_text,
        "non_dict_metadata_ids": bad_meta,
        "title_words": _summ(title_words),
        "abstract_words": _summ(text_words),
    }


def validate_queries(records: list[dict]) -> dict[str, Any]:
    errors: list[str] = []
    seen: Counter = Counter()
    label_counts: Counter = Counter()
    n_meta = 0
    for i, rec in enumerate(records):
        qid = rec.get("_id")
        if not isinstance(qid, str) or not qid.strip():
            errors.append(f"query record #{i}: _id must be a non-blank string, got {qid!r}")
            continue
        seen[qid] += 1
        if _is_blank(rec.get("text")):
            errors.append(f"query {qid}: blank or missing text")
        meta = rec.get("metadata", {})
        if not isinstance(meta, dict):
            errors.append(f"query {qid}: metadata must be a dict, got {type(meta).__name__}")
            continue
        if meta:
            n_meta += 1
            for annots in meta.values():  # informational only: NOT relevance labels
                for a in annots if isinstance(annots, list) else []:
                    label_counts[a.get("label", "<none>")] += 1
    dups = sorted(q for q, c in seen.items() if c > 1)
    if dups:
        errors.append(f"duplicate query _id values ({len(dups)}): {dups[:10]}")
    if errors:
        raise DataValidationError("Query validation failed:\n  " + "\n  ".join(errors[:20]))
    return {
        "n_queries": len(records),
        "n_unique_ids": len(seen),
        "n_with_metadata": n_meta,
        "metadata_annotation_label_counts_INFO_ONLY": dict(label_counts),
        "query_words": _summ([len(r["text"].split()) for r in records]),
    }


def validate_qrels(
    name: str,
    rows: list[tuple[str, str, int]],
    corpus_ids: set[str],
    query_ids: set[str],
) -> dict[str, Any]:
    """Hard errors: unknown corpus/query IDs, duplicate (query, doc) pairs, scores that are not positive."""
    errors: list[str] = []
    pairs = Counter((q, d) for q, d, _ in rows)
    dup_pairs = [p for p, c in pairs.items() if c > 1]
    missing_docs = sorted({d for _, d, _ in rows} - corpus_ids)
    missing_queries = sorted({q for q, _, _ in rows} - query_ids)
    if dup_pairs:
        errors.append(f"{name}: duplicate (query, doc) pairs: {dup_pairs[:5]}")
    if missing_docs:
        errors.append(f"{name}: corpus-ids not in corpus.jsonl ({len(missing_docs)}): {missing_docs[:10]}")
    if missing_queries:
        errors.append(f"{name}: query-ids not in queries.jsonl ({len(missing_queries)}): {missing_queries[:10]}")
    score_counts = Counter(s for _, _, s in rows)
    if any(s <= 0 for s in score_counts):
        errors.append(f"{name}: non-positive scores present: {dict(score_counts)}")
    if errors:
        raise DataValidationError("\n".join(errors))
    per_query = Counter(q for q, _, _ in rows)
    return {
        "n_rows": len(rows),
        "n_unique_queries": len(per_query),
        "n_unique_docs": len({d for _, d, _ in rows}),
        "score_counts": {str(k): v for k, v in score_counts.items()},
        "positives_per_query": _summ(list(per_query.values())),
        "positives_per_query_histogram": {str(k): v for k, v in sorted(Counter(per_query.values()).items())},
    }


def check_cross_qrels(train_rows, test_rows) -> dict[str, Any]:
    tq = {q for q, _, _ in train_rows}
    sq = {q for q, _, _ in test_rows}
    overlap = sorted(tq & sq)
    if overlap:
        raise DataValidationError(f"Official train and test share {len(overlap)} query IDs: {overlap[:10]}")
    return {"train_test_query_overlap": 0}


def _summ(values: list[int]) -> dict[str, float]:
    if not values:
        return {}
    s = sorted(values)
    return {
        "min": s[0],
        "median": statistics.median(s),
        "mean": round(statistics.fmean(s), 2),
        "p95": s[int(0.95 * (len(s) - 1))],
        "max": s[-1],
    }
