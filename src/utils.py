"""Small shared helpers: hashing, seeding, logging, JSON I/O."""
from __future__ import annotations

import hashlib
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file (RAM-safe for large files)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    """Seed python, NumPy and (if installed) torch. CPU determinism can still vary across versions."""
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is not required until Phase 2
        import torch  # type: ignore

        torch.manual_seed(seed)
    except ImportError:
        pass


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, sort_keys=True)
        fh.write("\n")


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def install_staged(stage_dir: Path, dest_dir: Path, force: bool = False) -> dict[str, str]:
    """Copy staged files into dest_dir. Existing files whose content would CHANGE are never overwritten
    unless force=True (raises FileExistsError naming them). Returns {name: created|unchanged|overwritten}."""
    import shutil

    stage_dir, dest_dir = Path(stage_dir), Path(dest_dir)
    conflicts = [f.name for f in stage_dir.iterdir()
                 if (dest_dir / f.name).exists() and sha256_file(dest_dir / f.name) != sha256_file(f)]
    if conflicts and not force:
        raise FileExistsError(f"Existing files would change: {conflicts}. Review, then re-run with --force.")
    dest_dir.mkdir(parents=True, exist_ok=True)
    status = {}
    for f in sorted(stage_dir.iterdir()):
        dest = dest_dir / f.name
        if dest.exists() and sha256_file(dest) == sha256_file(f):
            status[f.name] = "unchanged"
        else:
            status[f.name] = "overwritten" if dest.exists() else "created"
            shutil.copyfile(f, dest)
    return status
