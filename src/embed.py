"""Embedding utilities: model loading with provenance, batched float32 encoding, validated caching.
Heavy imports (torch / sentence-transformers) are lazy so the rest of the repo works without them."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .utils import get_logger, sha256_file, utc_now

log = get_logger("embed")


class CacheMismatch(Exception):
    """Cached embeddings do not match the requested model/config/corpus (rebuild required)."""


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, eps)).astype(np.float32, copy=False)


def load_sentence_model(name: str, revision: str | None, max_seq_length: int, device: str = "cpu"):
    """Download/resolve the model and return (model, provenance). Provenance carries the exact
    commit hash and the SHA-256 of the weights file."""
    try:
        import sentence_transformers
        import torch
        import transformers
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            f"Missing dependency ({exc.name}). Install with: python -m pip install -r requirements.txt"
        ) from exc
    local_dir = Path(snapshot_download(repo_id=name, revision=revision))
    model = SentenceTransformer(str(local_dir), device=device)
    model.max_seq_length = max_seq_length
    weights = next((p for p in (local_dir / "model.safetensors", local_dir / "pytorch_model.bin") if p.exists()), None)
    prov = {
        "model_name": name,
        "requested_revision": revision,
        "resolved_revision": local_dir.name,   # snapshots/<commit-hash>
        "weights_file": weights.name if weights else None,
        "weights_sha256": sha256_file(weights) if weights else None,
        "max_seq_length": max_seq_length,
        "embedding_dim": model.get_sentence_embedding_dimension(),
        "device": device,
        "versions": {"torch": torch.__version__, "sentence_transformers": sentence_transformers.__version__,
                     "transformers": transformers.__version__},
    }
    return model, prov


def encode_texts(encode_fn: Callable[[list[str]], np.ndarray], texts: list[str], chunk_size: int = 256,
                 normalize: bool = True) -> np.ndarray:
    """Encode in chunks (bounded memory), return float32 (n, dim), optionally L2-normalized."""
    chunks = []
    t0 = time.perf_counter()
    for i in range(0, len(texts), chunk_size):
        chunks.append(np.asarray(encode_fn(texts[i:i + chunk_size]), dtype=np.float32))
        done = min(i + chunk_size, len(texts))
        log.info("encoded %d/%d texts (%.1fs elapsed)", done, len(texts), time.perf_counter() - t0)
    emb = np.vstack(chunks) if chunks else np.zeros((0, 0), dtype=np.float32)
    return l2_normalize(emb) if normalize else emb


def make_encode_fn(model, batch_size: int) -> Callable[[list[str]], np.ndarray]:
    def fn(batch: list[str]) -> np.ndarray:
        return model.encode(batch, batch_size=batch_size, convert_to_numpy=True,
                            normalize_embeddings=False, show_progress_bar=False)
    return fn


def measure_truncation(tokenizer, texts: list[str], max_len: int) -> dict:
    """Exact count of documents whose tokenized length exceeds `max_len` (they get truncated)."""
    lengths = [len(tokenizer(t, add_special_tokens=True, truncation=False, verbose=False)["input_ids"])
               for t in texts]
    arr = np.array(lengths)
    return {"max_seq_length": max_len, "n_texts": len(texts), "n_truncated": int((arr > max_len).sum()),
            "fraction_truncated": round(float((arr > max_len).mean()), 4),
            "token_length_median": float(np.median(arr)), "token_length_p95": float(np.percentile(arr, 95)),
            "token_length_max": int(arr.max())}


# ---- caching ---------------------------------------------------------------------------------
def cache_key(corpus_hash: str, prov: dict, normalize: bool, batch_size: int) -> dict:
    """Fields that must match for a cache to be reusable (batch_size included only as provenance)."""
    return {"corpus_sha256": corpus_hash, "model_name": prov["model_name"],
            "resolved_revision": prov["resolved_revision"], "weights_sha256": prov["weights_sha256"],
            "max_seq_length": prov["max_seq_length"], "normalize": normalize,
            "embedding_dim": prov["embedding_dim"]}


def save_embeddings(emb: np.ndarray, key: dict, extra: dict, out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    npy = out_dir / f"{stem}.npy"
    np.save(npy, emb.astype(np.float32, copy=False))
    meta = {"cache_key": key, "extra": extra, "shape": list(emb.shape), "dtype": "float32",
            "npy_sha256": sha256_file(npy), "created_utc": utc_now()}
    (out_dir / f"{stem}.meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return npy


def load_embeddings(out_dir: Path, stem: str, expected_key: dict) -> np.ndarray:
    """Return cached embeddings or raise CacheMismatch / FileNotFoundError (caller rebuilds)."""
    npy, meta_p = out_dir / f"{stem}.npy", out_dir / f"{stem}.meta.json"
    if not npy.is_file() or not meta_p.is_file():
        raise FileNotFoundError(f"No cached embeddings at {npy}")
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    diffs = {k: (meta["cache_key"].get(k), v) for k, v in expected_key.items() if meta["cache_key"].get(k) != v}
    if diffs:
        raise CacheMismatch(f"Cache {npy.name} is stale; differing fields (cached, expected): {diffs}")
    if sha256_file(npy) != meta["npy_sha256"]:
        raise CacheMismatch(f"{npy.name} content hash does not match its meta file (corrupted?)")
    return np.load(npy)
