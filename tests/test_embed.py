import numpy as np
import pytest

from src.embed import CacheMismatch, cache_key, encode_texts, l2_normalize, load_embeddings, save_embeddings

# SYNTHETIC fake encoder: tests caching/chunking mechanics only, says nothing about MiniLM quality.
def fake_encode(batch):
    return np.array([[len(t), sum(map(ord, t)) % 7 + 1, 1.0] for t in batch], dtype=np.float64)

PROV = {"model_name": "m", "resolved_revision": "abc", "weights_sha256": "w", "max_seq_length": 256, "embedding_dim": 3}
TEXTS = ["alpha", "beta gamma", "delta", "epsilon zeta eta", "theta"]


def test_chunking_does_not_change_result_and_output_is_float32_unit_norm():
    a = encode_texts(fake_encode, TEXTS, chunk_size=2)
    b = encode_texts(fake_encode, TEXTS, chunk_size=100)
    assert a.dtype == np.float32 and np.allclose(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-6)


def test_zero_vector_does_not_produce_nan():
    assert not np.isnan(l2_normalize(np.zeros((2, 3)))).any()


def test_cache_roundtrip_and_invalidation(tmp_path):
    emb = encode_texts(fake_encode, TEXTS)
    key = cache_key("corpushash", PROV, True, 8)
    save_embeddings(emb, key, {}, tmp_path, "docs")
    assert np.array_equal(load_embeddings(tmp_path, "docs", key), emb)
    for change in ({"corpus_sha256": "other"}, {"resolved_revision": "zzz"}, {"max_seq_length": 128}, {"normalize": False}):
        with pytest.raises(CacheMismatch):
            load_embeddings(tmp_path, "docs", {**key, **change})


def test_corrupted_cache_detected(tmp_path):
    emb = encode_texts(fake_encode, TEXTS)
    key = cache_key("h", PROV, True, 8)
    save_embeddings(emb, key, {}, tmp_path, "docs")
    np.save(tmp_path / "docs.npy", emb + 1)
    with pytest.raises(CacheMismatch, match="hash"):
        load_embeddings(tmp_path, "docs", key)


def test_missing_cache_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_embeddings(tmp_path, "docs", {})
