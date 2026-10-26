"""Phase 6: packaged semantic-search service (exact cosine search over a frozen model's embeddings).

The package is self-contained and hash-verified:  model/  index/docs.faiss  index/document_ids.json  embeddings/docs.npy
documents.jsonl  service_manifest.json.  `SearchService.search(query, top_k)` returns ranked hits with rank, corpus_id,
title, snippet and similarity_score. similarity_score is a cosine similarity between L2-normalized embeddings; it is NOT a
probability that a paper is relevant or that a claim is true.
"""
from __future__ import annotations

import json
import numbers
import os
import re
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .embed import l2_normalize
from .utils import get_logger, read_json, sha256_file, utc_now, write_json

log = get_logger("search_service")
MANIFEST = "service_manifest.json"
_WS = re.compile(r"\s+")


class PackageIntegrityError(Exception):
    """A packaged file is missing or its SHA-256 differs from the manifest."""


class PackagingError(Exception):
    """A pre-condition for packaging (frozen selection, recorded test evaluation, matching hashes) is not met."""


# ---------------------------------------------------------------------------------------- result types
@dataclass(frozen=True)
class SearchHit:
    rank: int
    corpus_id: str
    title: str
    snippet: str
    similarity_score: float            # cosine similarity; not a probability

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SearchResponse:
    query: str                         # normalized query actually encoded
    status: str                        # ok | empty_query
    hits: list[SearchHit]
    top_k_requested: int | None
    top_k_effective: int
    warnings: list[str] = field(default_factory=list)
    latency_ms: dict[str, float] = field(default_factory=dict)


def make_snippet(text: str, n: int) -> str:
    text = _WS.sub(" ", text).strip()
    if len(text) <= n:
        return text
    cut = text[:n].rsplit(" ", 1)[0] or text[:n]
    return cut.rstrip(",;:.") + " …"


def normalize_query(q: str, max_chars: int) -> tuple[str, list[str]]:
    """Whitespace-collapse (incl. newlines/tabs) and drop NULs. Terms and case are NOT altered."""
    warnings = []
    q = _WS.sub(" ", q.replace("\x00", " ")).strip()
    if len(q) > max_chars:
        q, warnings = q[:max_chars].rstrip(), [f"query truncated to {max_chars} characters"]
    return q, warnings


# ---------------------------------------------------------------------------------------- service
class SearchService:
    def __init__(self, manifest: dict, encode_fn: Callable[[list[str]], np.ndarray], index, doc_ids: list[str],
                 docs: list[dict], startup_s: dict[str, float] | None = None):
        if not (index.ntotal == len(doc_ids) == len(docs)):
            raise PackageIntegrityError(f"index rows ({index.ntotal}), document IDs ({len(doc_ids)}) and documents ({len(docs)}) disagree")
        if [d["_id"] for d in docs] != doc_ids:
            raise PackageIntegrityError("documents.jsonl order differs from document_ids.json")
        s = manifest["service"]
        self.manifest, self._encode, self._index, self._ids, self._docs = manifest, encode_fn, index, doc_ids, docs
        self.default_top_k, self.max_top_k = int(s["default_top_k"]), int(s["max_top_k"])
        self.snippet_chars, self.max_query_chars = int(s["snippet_chars"]), int(s["max_query_chars"])
        self.startup_s = startup_s or {}
        self._lock = threading.Lock()          # single shared model/index: serialize calls

    # ---- construction -------------------------------------------------------------------------------------
    @classmethod
    def load(cls, package_dir: Path, verify: bool = True, encoder_factory: Callable[[Path, dict], Callable] | None = None) -> "SearchService":
        package_dir, t = Path(package_dir), {}
        mp = package_dir / MANIFEST
        if not mp.is_file():
            raise FileNotFoundError(f"{mp} not found. Build the package with `python scripts/build_search_package.py`.")
        manifest = read_json(mp)
        t0 = time.perf_counter()
        if verify:
            res = verify_package(package_dir)
            if not res["ok"]:
                raise PackageIntegrityError(f"package verification failed: missing={res['missing']} mismatched={res['mismatched']}")
        t["verify_hashes"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        encode_fn = (encoder_factory or default_encoder_factory)(package_dir / "model", manifest)
        t["load_model"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        import faiss

        index = faiss.read_index(str(package_dir / "index" / "docs.faiss"))
        doc_ids = json.loads((package_dir / "index" / "document_ids.json").read_text(encoding="utf-8"))
        t["load_index"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        docs = [json.loads(l) for l in (package_dir / "documents.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        t["load_documents"] = time.perf_counter() - t0
        if index.d != int(manifest["embedding_dim"]):
            raise PackageIntegrityError(f"index dim {index.d} != manifest {manifest['embedding_dim']}")
        t["total"] = sum(t.values())
        return cls(manifest, encode_fn, index, doc_ids, docs, t)

    # ---- API ----------------------------------------------------------------------------------------------
    def _resolve_top_k(self, top_k) -> tuple[int, list[str]]:
        if top_k is None:
            top_k = self.default_top_k
        if isinstance(top_k, bool) or not isinstance(top_k, numbers.Integral):
            raise TypeError(f"top_k must be an integer, got {type(top_k).__name__}")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        eff = min(int(top_k), self.max_top_k, self._index.ntotal)
        return eff, ([f"top_k clipped from {int(top_k)} to {eff} (service limit {self.max_top_k}, corpus {self._index.ntotal})"] if eff < top_k else [])

    def search_detailed(self, query: str, top_k: int | None = None) -> SearchResponse:
        if not isinstance(query, str):
            raise TypeError(f"query must be a string, got {type(query).__name__}")
        eff, warns = self._resolve_top_k(top_k)           # invalid top_k raises even for empty queries (caller bug)
        q, qw = normalize_query(query, self.max_query_chars)
        warns += qw
        if not q:
            return SearchResponse("", "empty_query", [], top_k, eff, warns + ["empty query: nothing to search"], {"total": 0.0})
        with self._lock:
            t0 = time.perf_counter()
            emb = l2_normalize(np.asarray(self._encode([q]), dtype=np.float32))
            t1 = time.perf_counter()
            scores, idxs = self._index.search(np.ascontiguousarray(emb), eff)
            t2 = time.perf_counter()
        hits = [SearchHit(rank=r + 1, corpus_id=self._ids[j], title=self._docs[j]["title"],
                          snippet=make_snippet(self._docs[j]["text"], self.snippet_chars), similarity_score=float(s))
                for r, (s, j) in enumerate(zip(scores[0], idxs[0])) if j >= 0]
        t3 = time.perf_counter()
        return SearchResponse(q, "ok", hits, top_k, eff, warns,
                              {"encode": (t1 - t0) * 1e3, "faiss_search": (t2 - t1) * 1e3, "format": (t3 - t2) * 1e3, "total": (t3 - t0) * 1e3})

    def search(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        """Ranked hits (empty list for an empty query). Raises TypeError/ValueError for invalid argument types/values."""
        return self.search_detailed(query, top_k).hits

    def info(self) -> dict[str, Any]:
        m = self.manifest
        return {"package": m["package_name"], "model": m["selected_model"], "n_documents": self._index.ntotal, "dim": self._index.d,
                "max_seq_length": m["max_seq_length"], "default_top_k": self.default_top_k, "max_top_k": self.max_top_k,
                "freeze_fingerprint": m["freeze_fingerprint"], "reported_metrics": m.get("reported_metrics"),
                "startup_s": self.startup_s}


def default_encoder_factory(model_dir: Path, manifest: dict) -> Callable[[list[str]], np.ndarray]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is required to load the packaged model: python -m pip install -r requirements.txt") from exc
    model = SentenceTransformer(str(model_dir), device="cpu")
    if int(model.max_seq_length) != int(manifest["max_seq_length"]):
        raise PackageIntegrityError(f"packaged model max_seq_length {model.max_seq_length} != manifest {manifest['max_seq_length']}")
    return lambda batch: model.encode(batch, batch_size=max(1, len(batch)), convert_to_numpy=True,
                                      normalize_embeddings=False, show_progress_bar=False)


# ---------------------------------------------------------------------------------------- packaging
def _file_table(root: Path, skip: set[str]) -> dict[str, dict]:
    return {p.relative_to(root).as_posix(): {"size": p.stat().st_size, "sha256": sha256_file(p)}
            for p in sorted(root.rglob("*")) if p.is_file() and p.relative_to(root).as_posix() not in skip}


def verify_package(package_dir: Path) -> dict[str, Any]:
    """Compare every file against the manifest (size + SHA-256). Extra files are reported but do not fail."""
    package_dir = Path(package_dir)
    t0 = time.perf_counter()
    manifest = read_json(package_dir / MANIFEST)
    missing, mismatched = [], []
    for rel, meta in manifest["files"].items():
        p = package_dir / rel
        if not p.is_file():
            missing.append(rel)
        elif p.stat().st_size != meta["size"] or sha256_file(p) != meta["sha256"]:
            mismatched.append(rel)
    extra = sorted(set(_file_table(package_dir, {MANIFEST})) - set(manifest["files"]))
    return {"ok": not missing and not mismatched, "missing": missing, "mismatched": mismatched, "extra": extra,
            "n_files": len(manifest["files"]), "seconds": time.perf_counter() - t0}


def build_package(cfg: dict, dest: Path | None = None) -> dict[str, Any]:
    """Package the FROZEN selected model. Pre-conditions: freeze intact, held-out test evaluation recorded, hashes equal the freeze.
    Never overwrites an existing package; builds in a temp directory and renames on success."""
    import shutil

    reports, root, proc = cfg["_abs"]["reports_dir"], cfg["_root"], cfg["_abs"]["processed_dir"]
    sel_p, test_p, ledger_p = reports / "model_selection.json", reports / "test_results.json", reports / "test_evaluation_ledger.json"
    for p in (sel_p, test_p, ledger_p):
        if not p.is_file():
            raise PackagingError(f"{p.name} not found: complete Phase 5 (validation + test) before packaging.")
    sel, test = read_json(sel_p), read_json(test_p)
    key = sel["selected"]
    if sel["status"] != "FROZEN" or test["status"] != "PASS" or test["selected_model"] != key:
        raise PackagingError(f"selection {sel['status']!r}, test {test['status']!r}, tested model {test['selected_model']!r} vs selected {key!r}")
    if not any(e["model"] == key and e["frozen_fingerprint"] == sel["fingerprint"] for e in json.loads(ledger_p.read_text(encoding="utf-8"))):
        raise PackagingError(f"no held-out test evaluation recorded for {key} under the current freeze")
    cand, frozen = cfg["candidates"][key], sel["candidates"][key]
    if cand["kind"] != "local":
        raise PackagingError("packaging a non-fine-tuned (pretrained) selection is not implemented; the selected model is local in this project")
    dest = Path(dest) if dest else cfg["_abs"]["service_dir"] / cfg["service"]["package_name"]
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f"{dest} already exists and is never overwritten. Move/delete it yourself, or change service.package_name.")
    model_src = root / cand["model_dir"]
    wfile = next((p for p in (model_src / "model.safetensors", model_src / "0_Transformer" / "model.safetensors") if p.exists()), None)
    emb_src = cfg["_abs"]["embeddings_dir"] / f"{key}_docs.npy"
    idx_src = cfg["_abs"]["indexes_dir"] / f"{key}_flatip.faiss"
    ids_src = cfg["_abs"]["indexes_dir"] / f"{key}_flatip.document_ids.json"
    checks = {"model weights": (wfile, frozen["weights_sha256"]), "document embeddings": (emb_src, frozen["embeddings_sha256"]),
              "FAISS index": (idx_src, frozen["index_sha256"])}
    for what, (p, want) in checks.items():
        if p is None or not p.is_file() or sha256_file(p) != want:
            raise PackagingError(f"{what} ({p}) does not match the frozen record; refusing to package.")
    doc_ids = read_json(proc / "document_ids.json")
    if read_json(ids_src) != doc_ids:
        raise PackagingError("index document-ID mapping differs from data/processed/document_ids.json")
    corpus = {r["_id"]: r for r in (json.loads(l) for l in (proc / "corpus_processed.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=dest.name + ".building-", dir=dest.parent))
    try:
        shutil.copytree(model_src, tmp / "model")
        (tmp / "embeddings").mkdir(); (tmp / "index").mkdir()
        shutil.copyfile(emb_src, tmp / "embeddings" / "docs.npy")
        shutil.copyfile(idx_src, tmp / "index" / "docs.faiss")
        shutil.copyfile(ids_src, tmp / "index" / "document_ids.json")
        with (tmp / "documents.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
            for d in doc_ids:
                fh.write(json.dumps({"_id": d, "title": corpus[d]["title"], "text": corpus[d]["text"]}, ensure_ascii=False) + "\n")
        import platform

        versions = {"python": platform.python_version()}
        for pkg in ("numpy", "faiss", "torch", "sentence_transformers", "transformers"):
            try:
                versions[pkg] = getattr(__import__(pkg), "__version__", None)
            except Exception:  # noqa: BLE001
                versions[pkg] = None
        vm = test["models"][key]
        manifest = {
            "package_version": 1, "package_name": cfg["service"]["package_name"], "created_utc": utc_now(),
            "selected_model": key, "label": cand["label"], "freeze_fingerprint": sel["fingerprint"],
            "weights_sha256": frozen["weights_sha256"], "embeddings_sha256": frozen["embeddings_sha256"], "index_sha256": frozen["index_sha256"],
            "corpus_sha256": sel["fingerprint_inputs"]["corpus_sha256"], "n_documents": len(doc_ids),
            "embedding_dim": None,
            "max_seq_length": cfg["model"]["max_seq_length"], "normalization": "L2 (queries and documents)", "metric": "cosine == inner product",
            "index_type": "IndexFlatIP (exact)", "service": {k: cfg["service"][k] for k in ("default_top_k", "max_top_k", "snippet_chars", "max_query_chars")},
            "reported_metrics": {"validation": vm["validation_metrics"], "heldout_test": vm["test_metrics"],
                                 "source": "artifacts/reports/test_results.json (Phase 5; one evaluation, frozen model)"},
            "library_versions_at_build": versions,
            "notice": "similarity_score is cosine similarity, not a probability of relevance or factual correctness.",
        }
        import faiss

        manifest["embedding_dim"] = int(faiss.read_index(str(tmp / "index" / "docs.faiss")).d)
        manifest["files"] = _file_table(tmp, {MANIFEST})
        write_json(manifest, tmp / MANIFEST)
        os.replace(tmp, dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)       # only our own half-built temp dir is removed
        raise
    return manifest


# ---------------------------------------------------------------------------------------- comparison / benchmark
def compare_rankings(ref: list[tuple[str, float]], got: list[tuple[str, float]], tol: float) -> dict[str, Any]:
    """Exact-or-tie-aware equivalence of two ranked lists. A rank swap is accepted only if the swapped scores differ by <= tol;
    documents present in only one list must score within tol of the other list's cutoff score (tie at the boundary)."""
    if len(ref) != len(got):
        return {"exact": False, "equivalent": False, "reason": f"lengths differ ({len(ref)} vs {len(got)})"}
    ri, gi = [d for d, _ in ref], [d for d, _ in got]
    if ri == gi:
        return {"exact": True, "equivalent": True, "reason": ""}
    pos_ok = all(abs(ref[i][1] - got[i][1]) <= tol for i in range(len(ref)) if ri[i] != gi[i])
    sr, sg = dict(ref), dict(got)
    edge_ok = all(abs(sr[d] - got[-1][1]) <= tol for d in set(ri) - set(gi)) and all(abs(sg[d] - ref[-1][1]) <= tol for d in set(gi) - set(ri))
    ok = pos_ok and edge_ok
    return {"exact": False, "equivalent": ok, "reason": "" if ok else "rank differences exceed score tolerance"}


def benchmark_service(service: SearchService, queries: list[str], top_ks: list[int], n_warmup: int, repeats: int) -> dict[str, Any]:
    """Per-query latency AFTER startup, single-threaded sequential calls. Startup timings are reported separately by the service."""
    out: dict[str, Any] = {}
    for _ in range(n_warmup):
        service.search_detailed(queries[0], top_ks[0])
    for k in top_ks:
        tot, enc, srch, fmt = [], [], [], []
        t_wall = time.perf_counter()
        for _ in range(repeats):
            for q in queries:
                lat = service.search_detailed(q, k).latency_ms
                tot.append(lat["total"]); enc.append(lat["encode"]); srch.append(lat["faiss_search"]); fmt.append(lat["format"])
        wall = time.perf_counter() - t_wall
        pct = lambda a, p: float(np.percentile(a, p))
        out[f"top_k={k}"] = {"n_calls": len(tot), "total_ms": {"mean": float(np.mean(tot)), "p50": pct(tot, 50), "p95": pct(tot, 95), "p99": pct(tot, 99), "max": float(max(tot))},
                             "encode_ms_mean": float(np.mean(enc)), "faiss_search_ms_mean": float(np.mean(srch)), "format_ms_mean": float(np.mean(fmt)),
                             "throughput_qps_sequential": len(tot) / wall}
    return out
