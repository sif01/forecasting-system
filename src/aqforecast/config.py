"""
config.py – Central configuration loader.

Merges configs/config.yaml (project-level settings that rarely change)
with params.yaml (tunables tracked by DVC for reproducibility).

Usage:
    from aqforecast.config import cfg, params
    print(cfg.location.latitude)
    print(params.xgboost.n_estimators)
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

# Repository root is three levels up from this file: src/aqforecast/config.py
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _REPO_ROOT / "configs" / "config.yaml"
_PARAMS_PATH = _REPO_ROOT / "params.yaml"


def _dict_to_namespace(d: dict[str, Any]) -> SimpleNamespace:
    """Recursively convert a nested dict to a SimpleNamespace for dot-access."""
    ns = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(ns, key, _dict_to_namespace(value))
        else:
            setattr(ns, key, value)
    return ns


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    with path.open("r") as fh:
        return yaml.safe_load(fh) or {}


# Eagerly load at import time so downstream modules get a simple object.
# Re-import or call load() explicitly if you need to reload during tests.
cfg: SimpleNamespace = _dict_to_namespace(_load_yaml(_CONFIG_PATH))
params: SimpleNamespace = _dict_to_namespace(_load_yaml(_PARAMS_PATH))

