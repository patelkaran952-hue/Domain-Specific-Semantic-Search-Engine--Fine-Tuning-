"""Exact cosine retrieval with FAISS IndexFlatIP over L2-normalized float32 vectors."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .utils import sha256_file


class IndexAlignmentError(Exception):
    pass


def _faiss():
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("faiss is not installed. Run: python -m pip install faiss-cpu "
                           "(verify a wheel exists for your Python version).") from exc
    return faiss


def check_unit_norm(emb: np.ndarray, tol: float = 1e-4) -> float:
    """Raise if any row norm deviates from 1 by more than tol; return the max deviation."""
    dev = float(np.max(np.abs(np.linalg.norm(emb, axis=1) - 1.0))) if len(emb) else 0.0
    if dev > tol:
        raise ValueError(f"Embeddings are not L2-normalized (max |norm-1| = {dev:.2e} > {tol}).")
    return dev


def build_index(emb: np.ndarray, tol: float = 1e-4):
    faiss = _faiss()
    emb = np.ascontiguousarray(emb, dtype=np.float32)
    if emb.ndim != 2 or emb.shape[0] == 0:
        raise ValueError(f"Expected a non-empty 2-D matrix, got shape {emb.shape}")
    check_unit_norm(emb, tol)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index


def save_index(index, doc_ids: list[str], out_dir: Path, stem: str, provenance: dict) -> dict:
    faiss = _faiss()
    out_dir.mkdir(parents=True, exist_ok=True)
    idx_path, map_path = out_dir / f"{stem}.faiss", out_dir / f"{stem}.document_ids.json"
    faiss.write_index(index, str(idx_path))
    map_path.write_text(json.dumps(doc_ids), encoding="utf-8")
    meta = {"ntotal": index.ntotal, "dim": index.d, "index_sha256": sha256_file(idx_path),
            "document_ids_sha256": sha256_file(map_path), "provenance": provenance}
    (out_dir / f"{stem}.index_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return meta


def load_index(out_dir: Path, stem: str, expected_doc_ids: list[str] | None = None):
    faiss = _faiss()
    idx_path, map_path = out_dir / f"{stem}.faiss", out_dir / f"{stem}.document_ids.json"
    for p in (idx_path, map_path):
        if not p.is_file():
            raise FileNotFoundError(f"Missing index artifact {p}. Rebuild with scripts/run_baseline.py")
    index = faiss.read_index(str(idx_path))
    doc_ids = json.loads(map_path.read_text(encoding="utf-8"))
    if index.ntotal != len(doc_ids):
        raise IndexAlignmentError(f"Index has {index.ntotal} rows but mapping has {len(doc_ids)} IDs.")
    if expected_doc_ids is not None and doc_ids != expected_doc_ids:
        raise IndexAlignmentError("Saved document-ID mapping differs from data/processed/document_ids.json.")
    return index, doc_ids


def search(index, doc_ids: list[str], query_emb: np.ndarray, top_k: int) -> list[list[tuple[str, float]]]:
    """Top-k (doc_id, cosine score) per query row, best first. top_k is clipped to the corpus size."""
    q = np.ascontiguousarray(query_emb, dtype=np.float32)
    if q.ndim == 1:
        q = q[None, :]
    k = max(1, min(int(top_k), index.ntotal))
    scores, idxs = index.search(q, k)
    return [[(doc_ids[j], float(s)) for s, j in zip(srow, irow) if j >= 0] for srow, irow in zip(scores, idxs)]
