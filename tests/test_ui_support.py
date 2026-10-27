import json
from pathlib import Path

import faiss
import numpy as np
import pytest

import src.search_service as S
from src import ui_support as ui
from src.compare import run_test_stage, run_validation_stage
from src.embed import l2_normalize
from src.search_service import PackageIntegrityError, SearchService, build_package
from src.utils import write_json
from tests.test_compare import DIMS, hash_encoder
from tests.test_search_service import add_service_cfg, encoder_factory, real_hash_loader
from tests.test_compare import make_cfg

# SYNTHETIC services/artifacts: tests the UI-support logic, not retrieval quality.


def fake_service(dim=32, n=30, titles=None):
    texts = [f"topic{i} alpha{i % 4} beta{i % 3}" for i in range(n)]
    docs = [{"_id": f"p{i:03d}", "title": (titles or {}).get(i, f"Title {i}"), "text": f"abstract of paper {i} " * 6} for i in range(n)]
    enc = hash_encoder(dim)
    emb = l2_normalize(enc(texts))
    index = faiss.IndexFlatIP(dim); index.add(emb)
    manifest = {"package_name": "fake", "selected_model": "m", "max_seq_length": 256, "freeze_fingerprint": "fp", "embedding_dim": dim,
                "service": {"default_top_k": 10, "max_top_k": 30, "snippet_chars": 60, "max_query_chars": 200}}
    return SearchService(manifest, enc, index, [d["_id"] for d in docs], docs)


def test_md_escape_neutralizes_markdown_and_math():
    s = ui.md_escape("Effect of *p53* on $x_1$ [cells] <b>bold</b> #1 `code`")
    for ch in "*$_[]<>#`":
        assert "\\" + ch in s
    assert ui.md_escape("plain text 123") == "plain text 123"


def test_run_search_outcomes():
    svc = fake_service()
    ok = ui.run_search(svc, "alpha1 topic3", 5)
    assert ok.level == "results" and len(ok.response.hits) == 5 and ok.messages == []
    for q in ("", "   "):
        o = ui.run_search(svc, q, 5)
        assert o.level == "empty" and o.messages[0][0] == "info" and o.response.hits == []
    bad = ui.run_search(svc, "alpha", 0)
    assert bad.level == "error" and bad.messages[0][0] == "error" and "top_k" in bad.messages[0][1]
    clipped = ui.run_search(svc, "alpha", 10**6)
    assert clipped.level == "results" and clipped.messages[0][0] == "warning"

    class Boom:
        def search_detailed(self, *a, **k):
            raise RuntimeError("index exploded")
    o = ui.run_search(Boom(), "x", 3)
    assert o.level == "error" and "RuntimeError" in o.messages[0][1]


def test_overlap_summary():
    a = fake_service(32).search("alpha1 topic3", 10)
    s = ui.overlap_summary(a, a)
    assert s["n_shared"] == 10 and s["same_top1"] is True
    b = fake_service(64).search("beta2 topic9", 10)
    assert 0 <= ui.overlap_summary(a, b)["n_shared"] <= 10 and ui.overlap_summary([], b)["same_top1"] is False


# ---- discovery + loading on a synthetic project ------------------------------------------------------------------------
def project(tmp_path, build=True):
    cfg = add_service_cfg(make_cfg(tmp_path), tmp_path)
    cfg["app"] = {"compare_enabled": True, "warmup_query": "warm-up query"}
    cfg["candidates"]["baseline"]["label"] = "Baseline"
    run_validation_stage(cfg, real_hash_loader(tmp_path))
    run_test_stage(cfg, real_hash_loader(tmp_path))
    if build:
        build_package(cfg)
    return cfg


def test_discover_models_offers_baseline_only_when_assets_exist(tmp_path):
    cfg = project(tmp_path)
    opts = ui.discover_models(cfg)
    assert [o.key for o in opts] == ["ft_hardneg_dense", "baseline"] and all(o.available for o in opts)
    assert opts[0].reported.get("heldout_test_ndcg@10") is not None
    (tmp_path / "artifacts/indexes/baseline_flatip.faiss").unlink()
    opts = ui.discover_models(cfg)
    assert opts[1].key == "baseline" and not opts[1].available and "run_baseline" in opts[1].reason
    cfg["app"]["compare_enabled"] = False
    assert [o.key for o in ui.discover_models(cfg)] == ["ft_hardneg_dense"]


def test_discover_models_without_package_or_freeze(tmp_path):
    cfg = project(tmp_path, build=False)
    first = ui.discover_models(cfg)[0]
    assert not first.available and "build_search_package" in first.reason
    (tmp_path / "artifacts/reports/model_selection.json").unlink()
    assert ui.discover_models(cfg) == []


def test_load_service_primary_and_unavailable(tmp_path, monkeypatch):
    cfg = project(tmp_path)
    monkeypatch.setattr(S, "default_encoder_factory", encoder_factory)
    svc = ui.load_service(cfg, "ft_hardneg_dense")
    assert svc.info()["n_documents"] == 60 and len(svc.search("topic12 alpha2", 3)) == 3
    with pytest.raises(FileNotFoundError, match="unknown model"):
        ui.load_service(cfg, "nope")
    (tmp_path / "artifacts/indexes/baseline_flatip.faiss").unlink()
    with pytest.raises(FileNotFoundError, match="not available"):
        ui.load_service(cfg, "baseline")


def baseline_loader(cfg, sha="w_baseline"):
    return hash_encoder(DIMS["baseline"]), {"weights_sha256": sha}


def test_load_baseline_service_verifies_frozen_hashes(tmp_path):
    cfg = project(tmp_path)
    svc = ui.load_baseline_service(cfg, model_loader=baseline_loader)
    hits = svc.search("topic5 alpha0 beta5 gamma2", 3)
    assert hits[0].corpus_id == "d005" and svc.get_document("d005")["title"] == "title 5"
    with pytest.raises(PackageIntegrityError, match="weights"):
        ui.load_baseline_service(cfg, model_loader=lambda c: baseline_loader(c, "tampered"))
    (tmp_path / "artifacts/embeddings/baseline_docs.npy").write_bytes(b"corrupt")
    with pytest.raises(PackageIntegrityError, match="embeddings"):
        ui.load_baseline_service(cfg, model_loader=baseline_loader)


def test_get_document_unknown_id_raises():
    with pytest.raises(KeyError):
        fake_service().get_document("zzz")
