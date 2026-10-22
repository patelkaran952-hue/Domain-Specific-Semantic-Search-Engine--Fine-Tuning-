"""Seeded, query-grouped train/validation split of the official training qrels."""
from __future__ import annotations

import math

import numpy as np


def split_query_ids(train_query_ids: list[str], val_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    """Deterministic split: sort IDs (as strings), permute with a seeded Generator, take the first
    round(val_fraction * n) as validation. Returns (train_ids, val_ids), each sorted."""
    ids = sorted(set(train_query_ids))
    if len(ids) < 2:
        raise ValueError("Need at least 2 unique queries to split.")
    n_val = max(1, min(len(ids) - 1, int(round(val_fraction * len(ids)))))
    perm = np.random.default_rng(seed).permutation(len(ids))
    val = sorted(ids[i] for i in perm[:n_val])
    train = sorted(ids[i] for i in perm[n_val:])
    return train, val


def partition_qrels(rows, train_ids: list[str], val_ids: list[str]):
    """Assign every qrel row to the partition holding its query. Raises if any row is orphaned."""
    t, v = set(train_ids), set(val_ids)
    if t & v:
        raise ValueError(f"Train/val query ID sets overlap: {sorted(t & v)[:5]}")
    train_rows, val_rows = [], []
    for row in rows:
        if row[0] in t:
            train_rows.append(row)
        elif row[0] in v:
            val_rows.append(row)
        else:
            raise ValueError(f"Qrel row {row} has a query ID in neither partition (labeled pair would be dropped).")
    return train_rows, val_rows


def verify_no_leakage(train_ids, val_ids, test_ids, train_rows, val_rows) -> dict:
    """Return a dict of gate results; every value must be True for the phase to pass."""
    t, v, s = set(train_ids), set(val_ids), set(test_ids)
    return {
        "train_val_disjoint": not (t & v),
        "train_test_disjoint": not (t & s),
        "val_test_disjoint": not (v & s),
        "train_rows_only_train_queries": all(r[0] in t for r in train_rows),
        "val_rows_only_val_queries": all(r[0] in v for r in val_rows),
        "no_query_straddles_partitions": not ({r[0] for r in train_rows} & {r[0] for r in val_rows}),
    }
