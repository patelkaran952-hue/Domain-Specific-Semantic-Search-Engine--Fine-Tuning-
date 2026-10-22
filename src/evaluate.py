"""Standard query-averaged ranking metrics with binary relevance.

Definitions (rank r is 1-based, rel_r in {0,1}, R = number of judged positives for the query):
  Recall@k    = |positives in top k| / R
  Precision@k = |positives in top k| / k
  MRR@k       = 1 / rank of the first positive within top k, else 0
  nDCG@k      = DCG@k / IDCG@k, DCG@k = sum_{r<=k} rel_r / log2(r + 1); IDCG@k uses min(R, k) ideal hits
  MAP@k       = (1 / min(R, k)) * sum_{r<=k} Precision@r * rel_r     (AP normalized by min(R, k))
Every labeled query is scored; a query with no run (or an empty one) scores 0 and is counted.
Metrics measure agreement with the *known judged positives* only (sparse qrels), not exhaustive relevance.
"""
from __future__ import annotations

import math
from typing import Iterable

METRIC_DEFINITIONS = {
    "recall@k": "|positives in top k| / R, averaged over queries",
    "precision@k": "|positives in top k| / k",
    "mrr@k": "reciprocal rank of first positive within top k, else 0",
    "ndcg@k": "binary-gain DCG with log2(rank+1) discount / ideal DCG with min(R,k) hits",
    "map@k": "sum_{r<=k} P@r*rel_r / min(R,k)",
}


def query_metrics(ranked: list[str], positives: set[str], ks=(1, 5, 10), ndcg_k=10, mrr_k=10,
                  precision_k=10, map_k=10) -> dict[str, float]:
    if not positives:
        raise ValueError("query_metrics requires at least one positive (labeled query).")
    R = len(positives)
    # de-duplicate while keeping order, so a repeated doc cannot be counted twice
    seen, ranking = set(), []
    for d in ranked:
        if d not in seen:
            seen.add(d)
            ranking.append(d)
    rel = [1 if d in positives else 0 for d in ranking]
    out: dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = sum(rel[:k]) / R
    out[f"precision@{precision_k}"] = sum(rel[:precision_k]) / precision_k
    out[f"mrr@{mrr_k}"] = next((1.0 / (i + 1) for i, r in enumerate(rel[:mrr_k]) if r), 0.0)
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel[:ndcg_k]))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(R, ndcg_k)))
    out[f"ndcg@{ndcg_k}"] = dcg / idcg
    hits, ap = 0, 0.0
    for i, r in enumerate(rel[:map_k]):
        if r:
            hits += 1
            ap += hits / (i + 1)
    out[f"map@{map_k}"] = ap / min(R, map_k)
    return out


def evaluate_run(run: dict[str, list[str]], qrels: dict[str, set[str]], **kw) -> dict:
    """Average metrics over ALL queries in `qrels`. Returns {'aggregate', 'per_query', 'n_queries', ...}."""
    per_query: dict[str, dict[str, float]] = {}
    missing = []
    for qid, positives in qrels.items():
        if qid not in run:
            missing.append(qid)
        per_query[qid] = query_metrics(run.get(qid, []), set(positives), **kw)
    extra = sorted(set(run) - set(qrels))
    names = next(iter(per_query.values())).keys() if per_query else []
    n = len(per_query)
    aggregate = {m: sum(pq[m] for pq in per_query.values()) / n for m in names} if n else {}
    return {
        "aggregate": aggregate,
        "per_query": per_query,
        "n_queries": n,
        "n_judgments": sum(len(v) for v in qrels.values()),
        "queries_missing_from_run": missing,
        "run_queries_without_labels_ignored": extra,
    }


def qrels_to_dict(rows: Iterable[tuple[str, str, int]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for q, d, s in rows:
        if s > 0:
            out.setdefault(q, set()).add(d)
    return out
