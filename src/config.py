"""YAML config loading with path resolution and validation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REQUIRED_SECTIONS = ("project", "paths", "raw_files", "preprocess", "split")


class ConfigError(Exception):
    pass


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upward from `start` (default: cwd) until configs/base.yaml is found."""
    here = Path(start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "configs" / "base.yaml").is_file():
            return candidate
    raise ConfigError(
        f"Could not find configs/base.yaml above {here}. Run from inside the repository "
        "(e.g. `cd scientific-semantic-search`)."
    )


def load_config(path: str | Path | None = None, root: Path | None = None) -> dict[str, Any]:
    """Load YAML config; adds cfg['_root'] and cfg['_abs'] (resolved absolute paths)."""
    root = Path(root).resolve() if root else find_repo_root()
    cfg_path = Path(path) if path else root / "configs" / "base.yaml"
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    if not cfg_path.is_file():
        raise ConfigError(f"Config file not found: {cfg_path}. Check the --config argument.")
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ConfigError(f"{cfg_path} is missing required section(s): {missing}")
    frac = cfg["split"].get("val_fraction")
    if not isinstance(frac, (int, float)) or not 0 < frac < 1:
        raise ConfigError(f"split.val_fraction must be in (0, 1); got {frac!r} in {cfg_path}")
    cfg["_root"] = root
    cfg["_config_path"] = cfg_path
    cfg["_abs"] = {k: root / v for k, v in cfg["paths"].items()}
    cfg["_raw"] = {k: cfg["_abs"]["raw_dir"] / v for k, v in cfg["raw_files"].items()}
    return cfg
