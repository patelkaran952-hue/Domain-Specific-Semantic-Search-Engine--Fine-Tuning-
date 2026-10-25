"""Positive-pair construction and hard-negative mining (training partition only).

Caveat repeated everywhere it matters: qrels are positive-only and sparse. A mined "negative" is merely an
UNJUDGED, highly ranked document; some are relevant (false negatives). Nothing here reads validation or test
judgments - callers pass in training qrels only.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


# ---------------------------------------------------------------------------------------- scoring
class BM25:
    """Okapi BM25 (Lucene-style non-negative idf) over a fixed document list; sparse and deterministic.
    Query term multiplicity is respected (repeated query terms count repeatedly), as in rank_bm25."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75, token_pattern: str = r"(?u)\b\w+\b"):
        from sklearn.feature_extraction.text import CountVectorizer
        from scipy import sparse

        self.vec = CountVectorizer(lowercase=True, token_pattern=token_pattern, dtype=np.float32)
        tf = self.vec.fit_transform(texts).tocsr().astype(np.float32)
        n_docs = tf.shape[0]
        df = np.asarray((tf > 0).sum(axis=0)).ravel()
        self.idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
        dl = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = float(dl.mean()) if n_docs else 1.0
        tf = tf.tocoo()
        denom = tf.data + k1 * (1.0 - b + b * dl[tf.row] / avgdl)
        w = self.idf[tf.col] * tf.data * (k1 + 1.0) / denom
        self.W = sparse.csr_matrix((w.astype(np.float32), (tf.row, tf.col)), shape=tf.shape)

    def score_matrix(self, queries: list[str]) -> np.ndarray:
        q = self.vec.transform(queries).astype(np.float32)          # term counts
        return np.asarray((q @ self.W.T).todense(), dtype=np.float32)   # (n_queries, n_docs)


def dense_score_matrix(q_emb: np.ndarray, doc_emb: np.ndarray) -> np.ndarray:
    """Exact inner product of L2-normalized vectors == cosine == what IndexFlatIP returns."""
    return (np.asarray(q_emb, dtype=np.float32) @ np.asarray(doc_emb, dtype=np.float32).T).astype(np.float32)


def stable_order(scores: np.ndarray, depth: int | None = None) -> np.ndarray:
    """Indices sorted by descending score, ties broken by ascending index (deterministic)."""
    order = np.lexsort((np.arange(len(scores)), -scores))
    return order if depth is None else order[:depth]


# ---------------------------------------------------------------------------------------- pairs
def build_positive_pairs(train_rows: Iterable[tuple[str, str, int]]) -> list[dict[str, str]]:
    """One row per judged (query, positive) pair, sorted by (query_id, positive_id)."""
    seen = set()
    for q, d, s in train_rows:
        if s > 0:
            seen.add((q, d))
    return [{"pair_id": f"{q}__{d}", "query_id": q, "positive_id": d} for q, d in sorted(seen)]


# ---------------------------------------------------------------------------------------- mining
def mine_query(qid: str, positives: set[str], scores: np.ndarray, doc_ids: list[str], *, candidate_depth: int,
               skip_top: int, pool_size: int, min_score: float, max_score_ratio: float | None,
               flag_score_ratio: float) -> dict[str, Any]:
    """Mine a pool of hard negatives for one training query from a full score row."""
    if not positives:
        raise ValueError(f"query {qid}: no positives; cannot mine against an unlabeled query")
    pos_idx = [i for i, d in enumerate(doc_ids) if d in positives]
    if len(pos_idx) != len(positives):
        raise ValueError(f"query {qid}: positive IDs missing from the document list")
    order_full = stable_order(scores)
    rank_of = np.empty(len(order_full), dtype=np.int64)
    rank_of[order_full] = np.arange(1, len(order_full) + 1)
    best_pos = max(pos_idx, key=lambda i: (scores[i], -i))
    best_pos_score = float(scores[best_pos])
    best_pos_rank = int(min(rank_of[i] for i in pos_idx))

    out_neg, ex = [], Counter()
    seen: set[str] = set()
    for idx in order_full[:candidate_depth]:
        d = doc_ids[idx]
        rank = int(rank_of[idx])
        if rank <= skip_top:
            ex["skipped_top_ranks"] += 1
        elif d in positives:
            ex["known_positive"] += 1
        elif d in seen:
            ex["duplicate"] += 1
        elif float(scores[idx]) <= min_score:
            ex["score_not_above_min"] += 1
        else:
            ratio = float(scores[idx]) / best_pos_score if best_pos_score > 0 else None
            if max_score_ratio is not None and ratio is not None and ratio >= max_score_ratio:
                ex["score_ratio_guard"] += 1
                continue
            seen.add(d)
            if len(out_neg) < pool_size:
                out_neg.append({"doc_id": d, "rank": rank, "score": round(float(scores[idx]), 6),
                                "score_ratio": None if ratio is None else round(ratio, 4),
                                "false_negative_risk_flag": bool(ratio is not None and ratio >= flag_score_ratio)})
    return {"query_id": qid, "positive_ids": sorted(positives), "best_positive_rank": best_pos_rank,
            "best_positive_score": round(best_pos_score, 6), "n_scanned": int(min(candidate_depth, len(order_full))),
            "excluded": dict(ex), "negatives": out_neg}


def sample_triplets(pairs: list[dict], mined: dict[str, dict], seed: int, per_pair: int = 1) -> list[dict]:
    """Seeded, deterministic (query, positive, negative) triplets; pairs whose pool is empty yield no triplet."""
    rng = np.random.default_rng(seed)
    out = []
    for p in pairs:                       # pairs are pre-sorted => deterministic RNG consumption
        pool = mined.get(p["query_id"], {}).get("negatives", [])
        if not pool:
            continue
        pick = rng.choice(len(pool), size=min(per_pair, len(pool)), replace=False)
        for j in sorted(int(x) for x in pick):
            n = pool[j]
            out.append({"pair_id": p["pair_id"], "query_id": p["query_id"], "positive_id": p["positive_id"],
                        "negative_id": n["doc_id"], "negative_rank": n["rank"], "negative_score": n["score"]})
    return out


# ---------------------------------------------------------------------------------------- I/O
def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing mining artifact {path}. Run scripts/mine_negatives.py first.")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_samples_md(path, mined, queries, titles, n, seed):
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(len(mined), size=min(n, len(mined)), replace=False))
    lines = ["# Hard-negative inspection samples", "",
             "> Negatives are UNJUDGED top-ranked documents, not verified irrelevant. Read for false negatives.", ""]
    for i in pick:
        m = mined[i]
        lines += [f"## Query {m['query_id']}", f"**Claim:** {queries[m['query_id']]}", "",
                  f"**Judged positive(s):** " + "; ".join(f"[{d}] {titles[d]}" for d in m["positive_ids"]),
                  f"(best positive rank {m['best_positive_rank']}, score {m['best_positive_score']})", ""]
        for n_ in m["negatives"]:
            flag = " ⚠ possible false negative" if n_["false_negative_risk_flag"] else ""
            lines.append(f"- rank {n_['rank']}, score {n_['score']}, ratio {n_['score_ratio']}: [{n_['doc_id']}] {titles[n_['doc_id']][:110]}{flag}")
        lines.append("")
    with path.open("w", encoding="utf-8", newline="\n") as fh:   # explicit LF: identical bytes on Windows and Linux
        fh.write("\n".join(lines))




# ---------------------------------------------------------------------------------------- gates + stats
def verify_mining_outputs(mining_dir: Path, source: str, train_rows, train_ids: list[str], val_ids: list[str],
                          test_ids: list[str], corpus_ids: set[str], pool_size: int) -> dict[str, bool]:
    """Re-read outputs FROM DISK and check every exit gate. Uses ID lists only for val/test (never their qrels)."""
    mined = read_jsonl_rows(mining_dir / f"hard_negatives_{source}.jsonl")
    trip = read_jsonl_rows(mining_dir / f"train_triplets_{source}.jsonl")
    pairs = read_jsonl_rows(mining_dir / "train_positive_pairs.jsonl")
    pos_of: dict[str, set[str]] = {}
    for q, d, s in train_rows:
        if s > 0:
            pos_of.setdefault(q, set()).add(d)
    tr, va, te = set(train_ids), set(val_ids), set(test_ids)
    qids = [m["query_id"] for m in mined]
    all_neg_ids = [n["doc_id"] for m in mined for n in m["negatives"]]
    nonempty = {m["query_id"] for m in mined if m["negatives"]}
    pair_set = {(p["query_id"], p["positive_id"]) for p in pairs}
    qrel_pairs = {(q, d) for q, ds in pos_of.items() for d in ds}
    return {
        "mined_queries_are_training_queries_only": set(qids) <= tr,
        "no_validation_or_test_queries": not (set(qids) & (va | te)) and not ({p["query_id"] for p in pairs} & (va | te)),
        "every_training_query_mined": set(qids) == tr and len(qids) == len(set(qids)),
        "no_known_positive_collisions": all(not ({n["doc_id"] for n in m["negatives"]} & pos_of.get(m["query_id"], set())) for m in mined),
        "no_duplicate_negatives_within_query": all(len({n["doc_id"] for n in m["negatives"]}) == len(m["negatives"]) for m in mined),
        "negatives_exist_in_corpus": set(all_neg_ids) <= corpus_ids,
        "pool_size_respected": all(len(m["negatives"]) <= pool_size for m in mined),
        "pairs_equal_train_qrels": pair_set == qrel_pairs,
        "triplets_pairs_in_train_qrels": all((t["query_id"], t["positive_id"]) in qrel_pairs for t in trip),
        "triplet_negative_never_its_positive": all(t["negative_id"] not in pos_of.get(t["query_id"], set()) for t in trip),
        "triplet_count_equals_pairs_with_pool": len(trip) == sum(1 for p in pairs if p["query_id"] in nonempty),
    }


def _q(a: list[float]) -> dict[str, float]:
    if not a:
        return {}
    x = np.asarray(a, dtype=np.float64)
    return {"min": float(x.min()), "p25": float(np.percentile(x, 25)), "median": float(np.median(x)),
            "p75": float(np.percentile(x, 75)), "p95": float(np.percentile(x, 95)), "max": float(x.max())}


def mining_stats(mined: list[dict], train_rows, pool_size: int, flag_ratio: float) -> dict[str, Any]:
    pos_of: dict[str, set[str]] = {}
    all_pos_train: set[str] = set()
    for q, d, s in train_rows:
        pos_of.setdefault(q, set()).add(d)
        all_pos_train.add(d)
    negs = [(m["query_id"], n) for m in mined for n in m["negatives"]]
    counts = Counter(n["doc_id"] for _, n in negs)
    ratios = [n["score_ratio"] for _, n in negs if n["score_ratio"] is not None]
    pos_ranks = [m["best_positive_rank"] for m in mined]
    n_q = len(mined)
    return {
        "n_queries": n_q,
        "n_negatives_in_pools": len(negs),
        "queries_with_full_pool": sum(1 for m in mined if len(m["negatives"]) == pool_size),
        "queries_with_empty_pool": sum(1 for m in mined if not m["negatives"]),
        "negative_score": _q([n["score"] for _, n in negs]),
        "best_positive_score": _q([m["best_positive_score"] for m in mined]),
        "negative_to_positive_score_ratio": _q(ratios),
        "flagged_false_negative_risk": {"threshold": flag_ratio, "n": sum(1 for _, n in negs if n["false_negative_risk_flag"]),
                                        "fraction": round(sum(1 for _, n in negs if n["false_negative_risk_flag"]) / max(1, len(negs)), 4)},
        "negative_rank": _q([n["rank"] for _, n in negs]),
        "train_query_best_positive_rank_diagnostics": {   # TRAIN queries, retriever diagnostic - not a validation metric
            "median": float(np.median(pos_ranks)),
            "frac_top1": round(sum(r == 1 for r in pos_ranks) / n_q, 4),
            "frac_top10": round(sum(r <= 10 for r in pos_ranks) / n_q, 4),
            "frac_top50": round(sum(r <= 50 for r in pos_ranks) / n_q, 4)},
        "unique_negative_docs": len(counts),
        "most_reused_negative_docs": counts.most_common(5),
        "negatives_that_are_positives_of_OTHER_train_queries": sum(1 for q, n in negs if n["doc_id"] in all_pos_train and n["doc_id"] not in pos_of[q]),
        "excluded_candidate_totals": dict(sum((Counter(m["excluded"]) for m in mined), Counter())),
    }
