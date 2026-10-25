"""Read-only inspection of fine-tuning run directories: how far did each attempt get, and why did it stop?"""
from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return str(n)


def _tree(path: Path, depth: int = 2, prefix: str = "    ") -> list[str]:
    lines = []
    for p in sorted(path.iterdir()):
        if p.is_dir():
            n = sum(1 for _ in p.rglob("*") if _.is_file())
            lines.append(f"{prefix}{p.name}/  ({n} files)")
            if depth > 1 and p.name != "checkpoints":
                lines += _tree(p, depth - 1, prefix + "  ")
            elif p.name == "checkpoints":
                lines += [f"{prefix}  {c.name}/" for c in sorted(p.iterdir()) if c.is_dir()]
        else:
            lines.append(f"{prefix}{p.name}  {_fmt_size(p.stat().st_size)}")
    return lines


def _load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"_unreadable": f"{type(exc).__name__}: {exc}"}


def describe_run(run_dir: Path, report_path: Path) -> list[str]:
    out = [f"=== {run_dir}"]
    if not run_dir.exists():
        return out + ["    (does not exist)"]
    out += ["  contents:"] + _tree(run_dir)
    out.append(f"  final report: {'FOUND (' + report_path.name + ')' if report_path.is_file() else 'MISSING -> run did not finish'}")
    failed = run_dir / "run_failed.json"
    if failed.is_file():
        d = _load(failed)
        out += [f"  run_failed.json: stage={d.get('stage')} | {d.get('exception_type')}: {str(d.get('message'))[:400]}",
                "  traceback (last 12 lines):"] + ["    " + l for l in str(d.get("traceback", "")).strip().splitlines()[-12:]]
    else:
        out.append("  run_failed.json: absent (attempt predates failure recording, or was killed / interrupted)")
    cfgp = run_dir / "training_config_resolved.json"
    if cfgp.is_file():
        c = _load(cfgp)
        out.append(f"  resolved config: rows={c.get('n_rows')} steps={c.get('steps')} smoke={c.get('smoke')} "
                   f"batch_size={c.get('trainer_args', {}).get('batch_size')}")
    hp = run_dir / "train_log_history.json"
    if hp.is_file():
        h = _load(hp)
        ev = [e for e in h if any(k.startswith("eval_") for k in e)] if isinstance(h, list) else []
        out.append(f"  train_log_history.json: {len(h)} entries, {len(ev)} evaluation entries (training reached the end of train())")
        if h:
            out.append(f"    last entry: {h[-1]}")
    else:
        out.append("  train_log_history.json: ABSENT -> trainer.train() did not complete")
    logp = run_dir / "train.log"
    if logp.is_file():
        tail = logp.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-8:]
        out += ["  train.log (last 8 lines):"] + ["    " + l[:220] for l in tail]
    final = run_dir / "final"
    out.append(f"  final model dir: {'present (' + str(len(list(final.iterdir()))) + ' files)' if final.is_dir() else 'absent'}")
    return out


def describe_environment() -> list[str]:
    out = ["=== installed versions"]
    for pkg in ("sentence-transformers", "transformers", "torch", "datasets", "accelerate", "huggingface-hub", "numpy", "faiss-cpu"):
        try:
            out.append(f"    {pkg}: {metadata.version(pkg)}")
        except metadata.PackageNotFoundError:
            out.append(f"    {pkg}: NOT INSTALLED")
    return out


def describe_pretrained_cache(path: Path) -> list[str]:
    if not path.is_file():
        return [f"=== {path.name}: absent"]
    d = _load(path)
    m = d.get("metrics", {})
    keys = [k for k in sorted(m) if k.endswith(("ndcg@10", "mrr@10", "recall@10", "recall@1"))]
    return [f"=== {path.name}: evaluator metrics of the PRETRAINED model (epoch 0), {d.get('seconds')}s to compute"] + \
           [f"    {k}: {m[k]:.4f}" for k in keys] + [f"    cache key: {d.get('key')}"]
