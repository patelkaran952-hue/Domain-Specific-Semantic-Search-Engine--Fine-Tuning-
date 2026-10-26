import json
from pathlib import Path

import numpy as np
import pytest

from src.compare import run_test_stage, run_validation_stage
from src.search_service import (MANIFEST, PackageIntegrityError, PackagingError, SearchService, benchmark_service, build_package,
                                compare_rankings, make_snippet, normalize_query, verify_package)
from src.service_validation import run_service_validation
from src.compare import Loaded
from src.utils import sha256_file, write_json
from tests.test_compare import DIMS, KEYS, hash_encoder, make_cfg

# SYNTHETIC project + hashing encoder: tests packaging/service mechanics, not model quality.


def encoder_factory(model_dir, manifest):
    return hash_encoder(int(manifest["embedding_dim"]))


def real_hash_loader(tmp: Path):
    """Like production: a local model's provenance hash is the SHA-256 of its actual weights file."""
    def loader(key, cand):
        wsha = sha256_file(tmp / cand["model_dir"] / "model.safetensors") if cand["kind"] == "local" else "w_baseline"
        prov = {"model_name": key, "resolved_revision": f"rev_{key}", "weights_sha256": wsha, "max_seq_length": 256, "embedding_dim": DIMS[key]}
        return Loaded(hash_encoder(DIMS[key]), prov)
    return loader


def add_service_cfg(cfg, tmp):
    cfg["_abs"]["service_dir"] = tmp / "artifacts" / "service"
    cfg["service"] = {"package_name": "pkg", "default_top_k": 5, "max_top_k": 20, "snippet_chars": 40, "max_query_chars": 200,
                      "verify_hashes_on_load": True}
    cfg["validation"] = {"tie_tolerance": 2e-5, "ndcg_tolerance": 1e-4, "score_tolerance": 1e-5, "n_score_samples": 4,
                         "example_queries": ["topic12 alpha2", "gamma1 beta3"]}
    cfg["benchmark"] = {"n_warmup": 1, "repeats": 1, "top_ks": [1, 5, 20]}
    return cfg


def built(tmp_path, run_test=True):
    cfg = add_service_cfg(make_cfg(tmp_path), tmp_path)
    run_validation_stage(cfg, real_hash_loader(tmp_path))
    if run_test:
        run_test_stage(cfg, real_hash_loader(tmp_path))
    return cfg


@pytest.fixture()
def svc_cfg(tmp_path):
    cfg = built(tmp_path)
    manifest = build_package(cfg)
    return cfg, manifest, cfg["_abs"]["service_dir"] / "pkg"


# ---- packaging --------------------------------------------------------------------------------------------------------
def test_package_requires_completed_phase5(tmp_path):
    cfg = built(tmp_path, run_test=False)
    with pytest.raises(PackagingError, match="Phase 5"):
        build_package(cfg)


def test_package_contents_manifest_and_verification(svc_cfg):
    cfg, m, pkg = svc_cfg
    for rel in ("model/model.safetensors", "embeddings/docs.npy", "index/docs.faiss", "index/document_ids.json", "documents.jsonl", MANIFEST):
        assert (pkg / rel).is_file(), rel
    assert m["selected_model"] in KEYS and m["embedding_dim"] == DIMS[m["selected_model"]] and m["n_documents"] == 60
    assert m["reported_metrics"]["heldout_test"] and "probability" in m["notice"]
    assert verify_package(pkg)["ok"]
    assert not list(pkg.parent.glob("*.building-*"))                       # no half-built temp dirs left behind


def test_package_never_overwrites_and_rejects_hash_drift(svc_cfg, tmp_path):
    cfg, m, pkg = svc_cfg
    with pytest.raises(SystemExit, match="never overwritten"):
        build_package(cfg)
    cfg2 = built(tmp_path / "b")
    (tmp_path / "b/artifacts/embeddings" / f"{json.loads((tmp_path / 'b/artifacts/reports/model_selection.json').read_text())['selected']}_docs.npy").write_bytes(b"corrupt")
    with pytest.raises(PackagingError, match="frozen record"):
        build_package(cfg2)
    assert not (tmp_path / "b/artifacts/service/pkg").exists() and not list((tmp_path / "b/artifacts/service").glob("*.building-*"))


def test_load_detects_tampering_and_missing_files(svc_cfg):
    cfg, m, pkg = svc_cfg
    SearchService.load(pkg, encoder_factory=encoder_factory)
    (pkg / "documents.jsonl").write_text("{}")
    with pytest.raises(PackageIntegrityError, match="mismatched"):
        SearchService.load(pkg, encoder_factory=encoder_factory)
    (pkg / "documents.jsonl").unlink()
    with pytest.raises(PackageIntegrityError, match="missing"):
        SearchService.load(pkg, encoder_factory=encoder_factory)
    with pytest.raises(FileNotFoundError, match="build_search_package"):
        SearchService.load(pkg.parent / "nope", encoder_factory=encoder_factory)


# ---- search behaviour -------------------------------------------------------------------------------------------------
def test_search_schema_ordering_and_self_retrieval(svc_cfg):
    cfg, m, pkg = svc_cfg
    svc = SearchService.load(pkg, encoder_factory=encoder_factory)
    hits = svc.search("topic7 alpha2 beta0 gamma1", 5)                      # exactly doc 7's indexed text
    assert [h.rank for h in hits] == [1, 2, 3, 4, 5] and hits[0].corpus_id == "d007" and hits[0].similarity_score == pytest.approx(1.0, abs=1e-5)
    assert all(a.similarity_score >= b.similarity_score for a, b in zip(hits, hits[1:])) and len({h.corpus_id for h in hits}) == 5
    assert set(hits[0].to_dict()) == {"rank", "corpus_id", "title", "snippet", "similarity_score"} and hits[0].title == "title 7"
    assert len(hits[0].snippet) <= 42
    assert svc.info()["n_documents"] == 60 and "startup_s" in svc.info()


def test_input_handling_contract(svc_cfg):
    cfg, m, pkg = svc_cfg
    svc = SearchService.load(pkg, encoder_factory=encoder_factory)
    for q in ("", "   ", "\n\t "):
        r = svc.search_detailed(q, 3)
        assert r.status == "empty_query" and r.hits == [] and r.warnings
    assert svc.search("") == []
    for bad_q in (None, 5, ["a"], b"bytes"):
        with pytest.raises(TypeError):
            svc.search(bad_q)
    for bad_k, exc in ((0, ValueError), (-1, ValueError), (2.5, TypeError), (True, TypeError), ("3", TypeError)):
        with pytest.raises(exc):
            svc.search("alpha", bad_k)
    big = svc.search_detailed("alpha", 10**6)
    assert big.top_k_effective == 20 and len(big.hits) == 20 and "clipped" in big.warnings[0]
    assert len(svc.search("alpha")) == 5                                    # default top_k
    assert np.int64(3) and len(svc.search("alpha", np.int64(3))) == 3       # numpy ints accepted
    long = svc.search_detailed("alpha " * 1000, 2)
    assert long.status == "ok" and any("truncated" in w for w in long.warnings)
    assert len(svc.search("β-catenin 研究 café", 2)) == 2 and len(svc.search("a\x00b", 2)) == 2
    assert [h.corpus_id for h in svc.search("alpha\n  beta\t gamma", 4)] == [h.corpus_id for h in svc.search("alpha beta gamma", 4)]


def test_helpers():
    assert normalize_query("  a \n b\t c ", 100) == ("a b c", [])
    q, w = normalize_query("x" * 50, 10)
    assert q == "x" * 10 and w
    assert make_snippet("one two three four five six", 12) == "one two …" and make_snippet("short", 50) == "short"
    assert make_snippet("a" * 30, 10).startswith("a" * 10)


def test_compare_rankings_tie_awareness():
    ref = [("a", 0.9), ("b", 0.8), ("c", 0.8), ("d", 0.5)]
    assert compare_rankings(ref, ref, 1e-5)["exact"]
    swapped = [("a", 0.9), ("c", 0.8), ("b", 0.8), ("d", 0.5)]
    r = compare_rankings(ref, swapped, 1e-5)
    assert not r["exact"] and r["equivalent"]                                # swap between tied scores is allowed
    wrong = [("b", 0.8), ("a", 0.9), ("c", 0.8), ("d", 0.5)]
    assert not compare_rankings(ref, wrong, 1e-5)["equivalent"]              # swap of clearly different scores is not
    boundary = [("a", 0.9), ("b", 0.8), ("c", 0.8), ("e", 0.5)]
    assert compare_rankings(ref, boundary, 1e-5)["equivalent"]               # tie at the cutoff
    assert not compare_rankings(ref, [("a", 0.9), ("b", 0.8), ("c", 0.8), ("e", 0.1)], 1e-5)["equivalent"]
    assert not compare_rankings(ref, ref[:3], 1e-5)["equivalent"]


# ---- validation + benchmark -------------------------------------------------------------------------------------------
def test_service_validation_passes_end_to_end_and_writes_report(svc_cfg):
    cfg, m, pkg = svc_cfg
    r = run_service_validation(cfg, pkg, encoder_factory)
    assert r["status"] == "PASS", [k for k, v in r["gates"].items() if not v]
    rc = r["reload_consistency"]
    assert rc["not_equivalent"] == 0 and rc["exact"] + rc["tie_aware_equivalent_not_exact"] == rc["n_queries"] == 6
    assert r["end_to_end_metric"]["abs_diff"] <= 1e-4 and r["score_check_max_abs_diff"] <= 1e-5
    assert set(r["latency_after_startup"]) == {"top_k=1", "top_k=5", "top_k=20"} and "total" in r["startup_s_excluded_from_latency"]
    assert (cfg["_abs"]["reports_dir"] / "search_service_report.json").is_file()
    assert set(r["example_queries"]) == {"topic12 alpha2", "gamma1 beta3"}


def test_service_validation_fails_when_reference_rankings_differ(svc_cfg):
    cfg, m, pkg = svc_cfg
    p = cfg["_abs"]["reports_dir"] / f"val_run_top100_{m['selected_model']}.json"
    ref = json.loads(p.read_text())
    q = sorted(ref)[0]
    ref[q] = list(reversed(ref[q]))
    p.write_text(json.dumps(ref))
    r = run_service_validation(cfg, pkg, encoder_factory)
    assert r["status"] == "FAIL" and r["gates"]["reloaded_rankings_match_phase5_run_tie_aware"] is False
    assert r["reload_consistency"]["not_equivalent"] >= 1


def test_benchmark_service_structure(svc_cfg):
    cfg, m, pkg = svc_cfg
    svc = SearchService.load(pkg, encoder_factory=encoder_factory)
    b = benchmark_service(svc, ["alpha", "beta gamma", "topic3"], [1, 5], n_warmup=1, repeats=2)
    assert set(b) == {"top_k=1", "top_k=5"} and b["top_k=5"]["n_calls"] == 6
    t = b["top_k=1"]["total_ms"]
    assert t["p50"] <= t["p95"] <= t["max"] and b["top_k=1"]["throughput_qps_sequential"] > 0
