"""Sentence-Transformers / transformers glue for Phase 4 (the ONLY version-sensitive code).

Written against the documented SentenceTransformerTrainer API (v3+). Import locations moved in newer releases
(e.g. sentence_transformers.sentence_transformer.losses), so classes are resolved through fallback chains and
keyword arguments are filtered by each callable's real signature. A preflight evaluation in src/train.py
verifies metric keys BEFORE any long computation. Not executable in the authoring sandbox (no torch / HF access).
"""
from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any

import numpy as np

from .embed import load_sentence_model
from .train import TrainResult
from .utils import get_logger

log = get_logger("train_backend")


def _first_attr(candidates: list[tuple[str, str]], what: str):
    """Return the first importable `module.attr`; error lists everything tried."""
    errs = []
    for mod, attr in candidates:
        try:
            return getattr(importlib.import_module(mod), attr)
        except (ImportError, AttributeError) as exc:
            errs.append(f"{mod}.{attr}: {exc}")
    raise ImportError(f"Could not import {what}. Tried: " + " | ".join(errs) +
                      ". Install with: python -m pip install \"sentence-transformers[train]\"")


def _supported_kwargs(fn, kwargs: dict, critical: frozenset = frozenset()) -> dict:
    """Keep only kwargs the callable accepts. Dropped optional args are logged; dropping a CRITICAL arg
    (one that changes what is trained/measured) raises instead of silently altering behavior."""
    params = inspect.signature(fn).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    dropped = sorted(k for k in kwargs if k not in params)
    bad = [k for k in dropped if k in critical]
    if bad:
        raise TypeError(f"Installed library does not accept required argument(s) {bad} for {getattr(fn, '__qualname__', fn)}. "
                        "Upgrade (python -m pip install -U \"sentence-transformers[train]\" transformers) or tell the assistant "
                        "the installed versions so the glue can be adapted.")
    if dropped:
        log.warning("Ignoring unsupported optional argument(s) %s for %s", dropped, getattr(fn, "__qualname__", fn))
    return {k: v for k, v in kwargs.items() if k in params}


EVAL_CRITICAL = frozenset({"queries", "corpus", "relevant_docs", "name", "ndcg_at_k", "score_functions"})
TRAIN_CRITICAL = frozenset({"output_dir", "num_train_epochs", "per_device_train_batch_size", "learning_rate", "warmup_steps",
                            "seed", "batch_sampler", "eval_strategy", "save_strategy", "load_best_model_at_end",
                            "metric_for_best_model", "greater_is_better"})


class SentenceTransformersBackend:
    def load_model(self, name: str, revision: str | None, max_seq_length: int):
        return load_sentence_model(name, revision, max_seq_length)

    def encode(self, model: Any, texts: list[str]) -> np.ndarray:
        return np.asarray(model.encode(texts, batch_size=4, convert_to_numpy=True, normalize_embeddings=False,
                                       show_progress_bar=False), dtype=np.float32)

    def max_seq_length(self, model: Any) -> int:
        return int(model.max_seq_length)

    def versions(self) -> dict:
        import sentence_transformers, torch, transformers
        out = {"torch": torch.__version__, "transformers": transformers.__version__,
               "sentence_transformers": sentence_transformers.__version__}
        for pkg in ("datasets", "accelerate"):
            try:
                out[pkg] = importlib.import_module(pkg).__version__
            except Exception:  # noqa: BLE001
                out[pkg] = None
        return out

    # ---- evaluation ---------------------------------------------------------------------------------
    def make_evaluator(self, queries: dict, corpus: dict, relevant: dict, cfg: dict):
        ev_cls = _first_attr([("sentence_transformers.sentence_transformer.evaluation", "InformationRetrievalEvaluator"),
                              ("sentence_transformers.evaluation", "InformationRetrievalEvaluator")], "InformationRetrievalEvaluator")
        cos_sim = _first_attr([("sentence_transformers.util", "cos_sim")], "cos_sim")
        kwargs = dict(queries=queries, corpus=corpus, relevant_docs=relevant, name="val",
                      accuracy_at_k=[1, 5, 10], precision_recall_at_k=[1, 5, 10], mrr_at_k=[10], ndcg_at_k=[10], map_at_k=[10],
                      score_functions={"cosine": cos_sim}, main_score_function="cosine",
                      batch_size=16, show_progress_bar=False, write_csv=False)
        return ev_cls(**_supported_kwargs(ev_cls.__init__, kwargs, EVAL_CRITICAL))

    def evaluate(self, evaluator: Any, model: Any) -> dict[str, float]:
        res = evaluator(model)
        if not isinstance(res, dict):
            raise TypeError(f"Evaluator returned {type(res).__name__}, expected a dict of metrics (sentence-transformers >= 3).")
        return {k: float(v) for k, v in res.items()}

    def primary_metric(self, evaluator: Any, cfg: dict) -> str:
        pm = getattr(evaluator, "primary_metric", None)
        return pm if isinstance(pm, str) and pm else "val_cosine_ndcg@10"

    # ---- training -------------------------------------------------------------------------------------
    def train(self, model: Any, data: dict[str, list[str]], evaluator: Any, args: dict, run_dir: Path) -> TrainResult:
        from datasets import Dataset
        from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
        losses = [("sentence_transformers.sentence_transformer.losses", "MultipleNegativesRankingLoss"),
                  ("sentence_transformers.losses", "MultipleNegativesRankingLoss")]
        loss_cls = _first_attr(losses, "MultipleNegativesRankingLoss")
        ds = Dataset.from_dict(data)                           # column order == loss input order (anchor, positive[, negative])
        targs = dict(
            output_dir=str(run_dir / "checkpoints"), num_train_epochs=args["epochs"],
            per_device_train_batch_size=args["batch_size"], per_device_eval_batch_size=args["batch_size"],
            learning_rate=args["learning_rate"], warmup_steps=args["warmup_steps"], weight_decay=args["weight_decay"],
            seed=args["seed"], data_seed=args["seed"], dataloader_num_workers=args["num_workers"],
            batch_sampler=args["batch_sampler"],                # "no_duplicates" == BatchSamplers.NO_DUPLICATES
            use_cpu=True, fp16=False, bf16=False, report_to="none",
            eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
            load_best_model_at_end=True, metric_for_best_model=args["metric_for_best_model"], greater_is_better=True,
            logging_steps=args["logging_steps"], disable_tqdm=True,
        )
        training_args = SentenceTransformerTrainingArguments(
            **_supported_kwargs(SentenceTransformerTrainingArguments.__init__, targs, TRAIN_CRITICAL))
        trainer = SentenceTransformerTrainer(model=model, args=training_args, train_dataset=ds,
                                             loss=loss_cls(model), evaluator=evaluator)
        out = trainer.train()
        return TrainResult(model=trainer.model, log_history=list(trainer.state.log_history),
                           global_step=int(trainer.state.global_step),
                           train_runtime_s=float(out.metrics.get("train_runtime", 0.0)),
                           best_checkpoint=trainer.state.best_model_checkpoint)

    def save(self, model: Any, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (getattr(model, "save_pretrained", None) or model.save)(str(path))

    def load_saved(self, path: Path) -> Any:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(str(path), device="cpu")
