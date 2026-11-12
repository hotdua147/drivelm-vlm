"""Latency and memory measurement helpers (torch optional)."""
from __future__ import annotations

import time
from contextlib import contextmanager


def _torch():
    try:
        import torch  # noqa: F401
        return torch
    except Exception:  # pragma: no cover
        return None


def reset_peak():
    t = _torch()
    if t is not None and t.cuda.is_available():
        t.cuda.reset_peak_memory_stats()


def peak_vram_gb():
    t = _torch()
    if t is not None and t.cuda.is_available():
        return t.cuda.max_memory_allocated() / 1024 ** 3
    return None


@contextmanager
def timer(out: dict, key: str = "latency_s"):
    t = _torch()
    if t is not None and t.cuda.is_available():
        t.cuda.synchronize()
    start = time.perf_counter()
    yield
    if t is not None and t.cuda.is_available():
        t.cuda.synchronize()
    out[key] = time.perf_counter() - start
