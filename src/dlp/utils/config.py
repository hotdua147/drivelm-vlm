"""YAML configs with ``_base_`` inheritance (paths relative to the including file)."""
from __future__ import annotations

import copy
import os
from typing import Any, Dict

import yaml


def deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    bases = cfg.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]
    merged: Dict[str, Any] = {}
    for b in bases:
        merged = deep_merge(merged, load_config(os.path.normpath(os.path.join(os.path.dirname(path), b))))
    return deep_merge(merged, cfg)


def set_dotted(cfg: Dict[str, Any], dotted: str, value: str) -> None:
    """CLI override ``a.b.c=value`` (value parsed as YAML)."""
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = yaml.safe_load(value)
