"""Scene-level split and subset labels (city, day/night/rain)."""
from __future__ import annotations

import hashlib
import re


def scene_split(scene_token: str, val_frac: float = 0.1, seed: int = 0) -> str:
    """Deterministic, order-independent split: hash(seed, scene) -> [0, 1)."""
    h = hashlib.sha1(f"{seed}:{scene_token}".encode()).hexdigest()
    return "val" if int(h[:8], 16) / 0xFFFFFFFF < val_frac else "train"


def city_of(location: str) -> str:
    loc = (location or "").lower()
    if loc.startswith("boston"):
        return "boston"
    if loc.startswith("singapore"):
        return "singapore"
    return loc or "unknown"


def condition_of(scene_description: str) -> str:
    d = (scene_description or "").lower()
    if re.search(r"\bnight\b", d):
        return "night"
    if re.search(r"\brain", d):
        return "rain"
    return "day"
