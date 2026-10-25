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


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}. Check the --config argument.")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(path: str | Path | list | tuple | None = None, root: Path | None = None) -> dict[str, Any]:
    """Load configs/base.yaml and deep-merge zero or more override YAMLs (in order) on top.
    Adds cfg['_root'], cfg['_abs'] (resolved paths) and cfg['_raw'] (raw data paths)."""
    root = Path(root).resolve() if root else find_repo_root()
    base_path = root / "configs" / "base.yaml"
    cfg = _read_yaml(base_path)
    paths = [] if path is None else ([path] if isinstance(path, (str, Path)) else list(path))
    cfg_path = base_path
    for item in paths:
        cfg_path = Path(item) if Path(item).is_absolute() else root / item
        if cfg_path.resolve() != base_path.resolve():
            cfg = _deep_merge(cfg, _read_yaml(cfg_path))
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
