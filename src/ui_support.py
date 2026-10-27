"""UI-independent logic for the Streamlit app (so it is unit-testable without a browser):
model discovery, service loading (fine-tuned package or pretrained baseline), safe search wrapper, text escaping, list overlap."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import load_config
from .embed import l2_normalize
from .search_service import MANIFEST, PackageIntegrityError, SearchHit, SearchResponse, SearchService
from .utils import read_json, sha256_file

CONFIGS = ["configs/baseline.yaml", "configs/compare.yaml", "configs/service.yaml", "configs/app.yaml"]
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+!|<>~$&])")


def get_config() -> dict:
    """Config stacked from the repository root (independent of the current working directory)."""
    return load_config(CONFIGS, root=Path(__file__).resolve().parents[1])


def md_escape(text: str) -> str:
    """Escape Markdown/LaTeX-significant characters so titles/abstracts render literally (e.g. '$' would start math, '*' emphasis)."""
    return _MD_SPECIAL.sub(r"\\\1", text)


@dataclass(frozen=True)
class ModelOption:
    key: str
    label: str
    available: bool
    reason: str = ""
    reported: dict = field(default_factory=dict)


def _read_optional(path: Path) -> dict | None:
    return read_json(path) if path.is_file() else None


def discover_models(cfg: dict) -> list[ModelOption]:
    """Primary = the packaged (frozen, selected) model. Baseline is offered ONLY if its embeddings/index exist and match the freeze record."""
    reports = cfg["_abs"]["reports_dir"]
    sel, test = _read_optional(reports / "model_selection.json"), _read_optional(reports / "test_results.json")
    if not sel or sel.get("status") != "FROZEN":
        return []
    key = sel["selected"]
    pkg = cfg["_abs"]["service_dir"] / cfg["service"]["package_name"]
    rep = lambda k: ({"heldout_test_ndcg@10": test["models"][k]["test_metrics"]["ndcg@10"]} if test and k in test.get("models", {}) else {})
    opts = [ModelOption(key, cfg["candidates"][key]["label"], (pkg / MANIFEST).is_file(),
                        "" if (pkg / MANIFEST).is_file() else f"package not built ({pkg}); run scripts/build_search_package.py", rep(key))]
    if key != "baseline" and cfg["app"]["compare_enabled"]:
        idx, emb = cfg["_abs"]["indexes_dir"] / "baseline_flatip.faiss", cfg["_abs"]["embeddings_dir"] / "baseline_docs.npy"
        ids = cfg["_abs"]["indexes_dir"] / "baseline_flatip.document_ids.json"
        have = all(p.is_file() for p in (idx, emb, ids))
        opts.append(ModelOption("baseline", cfg["candidates"]["baseline"]["label"], have,
                                "" if have else "baseline index/embeddings not found (run scripts/run_baseline.py)", rep("baseline")))
    return opts


def load_baseline_service(cfg: dict, model_loader: Callable | None = None, verify: bool = True) -> SearchService:
    """Pretrained-baseline service from the Phase 2/5 artifacts (no model copy). Weights, embeddings and index hashes must equal the freeze record."""
    import faiss

    reports, proc = cfg["_abs"]["reports_dir"], cfg["_abs"]["processed_dir"]
    sel = read_json(reports / "model_selection.json")
    frozen = sel["candidates"]["baseline"]
    emb_p, idx_p = cfg["_abs"]["embeddings_dir"] / "baseline_docs.npy", cfg["_abs"]["indexes_dir"] / "baseline_flatip.faiss"
    ids_p = cfg["_abs"]["indexes_dir"] / "baseline_flatip.document_ids.json"
    if verify:
        for what, p, want in (("baseline embeddings", emb_p, frozen["embeddings_sha256"]), ("baseline index", idx_p, frozen["index_sha256"])):
            if sha256_file(p) != want:
                raise PackageIntegrityError(f"{what} {p.name} does not match the frozen record")
    doc_ids = json.loads(ids_p.read_text(encoding="utf-8"))
    if doc_ids != read_json(proc / "document_ids.json"):
        raise PackageIntegrityError("baseline document-ID mapping differs from data/processed/document_ids.json")
    docs_by_id = {r["_id"]: r for r in (json.loads(l) for l in (proc / "corpus_processed.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    docs = [{"_id": d, "title": docs_by_id[d]["title"], "text": docs_by_id[d]["text"]} for d in doc_ids]
    if model_loader is None:
        from .embed import load_sentence_model, make_encode_fn

        mc = cfg["model"]
        try:                                              # offline-first: the snapshot is already in the local HF cache
            model, prov = load_sentence_model(mc["name"], mc["revision"], mc["max_seq_length"], local_files_only=True)
        except Exception:  # noqa: BLE001
            model, prov = load_sentence_model(mc["name"], mc["revision"], mc["max_seq_length"])
        encode_fn = lambda batch: model.encode(batch, batch_size=max(1, len(batch)), convert_to_numpy=True,
                                               normalize_embeddings=False, show_progress_bar=False)
    else:
        encode_fn, prov = model_loader(cfg)
    if verify and prov["weights_sha256"] != frozen["weights_sha256"]:
        raise PackageIntegrityError("pretrained baseline weights differ from the frozen record")
    index = faiss.read_index(str(idx_p))
    manifest = {"package_name": "baseline (pretrained, unpackaged)", "selected_model": "baseline", "label": cfg["candidates"]["baseline"]["label"],
                "max_seq_length": cfg["model"]["max_seq_length"], "freeze_fingerprint": sel["fingerprint"], "embedding_dim": int(index.d),
                "service": {k: cfg["service"][k] for k in ("default_top_k", "max_top_k", "snippet_chars", "max_query_chars")},
                "reported_metrics": None}
    return SearchService(manifest, encode_fn, index, doc_ids, docs, {})


def load_service(cfg: dict, key: str, verify: bool = True, warmup: bool = True) -> SearchService:
    """Load one service. NOT cached here: the Streamlit app wraps this in st.cache_resource so each model initializes once per process."""
    opts = {o.key: o for o in discover_models(cfg)}
    if key not in opts or not opts[key].available:
        raise FileNotFoundError(f"model {key!r} is not available: {opts[key].reason if key in opts else 'unknown model'}")
    if key == "baseline":
        svc = load_baseline_service(cfg, verify=verify)
    else:
        svc = SearchService.load(cfg["_abs"]["service_dir"] / cfg["service"]["package_name"], verify=verify and cfg["service"]["verify_hashes_on_load"])
    if warmup:
        svc.search(cfg["app"]["warmup_query"], 1)
    return svc


@dataclass
class UiOutcome:
    level: str                       # "results" | "empty" | "error"
    response: SearchResponse | None
    messages: list[tuple[str, str]]  # (streamlit level: info|warning|error, text)


def run_search(svc: SearchService, query: str, top_k: int) -> UiOutcome:
    """Never raises to the UI: invalid input and unexpected failures become messages."""
    try:
        r = svc.search_detailed(query, top_k)
    except (TypeError, ValueError) as exc:
        return UiOutcome("error", None, [("error", f"Invalid request: {exc}")])
    except Exception as exc:  # noqa: BLE001
        return UiOutcome("error", None, [("error", f"Search failed ({type(exc).__name__}): {exc}")])
    msgs = [("warning", w) for w in r.warnings]
    if r.status == "empty_query":
        return UiOutcome("empty", r, [("info", "Type a claim or topic to search the 5,183 SciFact abstracts.")])
    if not r.hits:
        return UiOutcome("empty", r, msgs + [("info", "No results found for this query.")])
    return UiOutcome("results", r, msgs)


def overlap_summary(a: list[SearchHit], b: list[SearchHit]) -> dict[str, Any]:
    ra, rb = {h.corpus_id: h.rank for h in a}, {h.corpus_id: h.rank for h in b}
    shared = [d for d in ra if d in rb]
    return {"k_a": len(a), "k_b": len(b), "n_shared": len(shared), "shared_ids": shared, "rank_in_a": ra, "rank_in_b": rb,
            "same_top1": bool(a and b and a[0].corpus_id == b[0].corpus_id)}
