"""Tag matching (Hungarian), grounding and hallucination against projected nuScenes boxes."""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..data.linking import dist_to_box, point_in_box


def match_tags(pred: Sequence[dict], gt: Sequence[dict], tau: float) -> dict:
    """One-to-one matching per camera; a pair counts if its pixel distance <= tau.

    ``pred`` / ``gt``: dicts with ``cam``, ``x``, ``y``. Returns tp/fp/fn and pairs.
    """
    tp, pairs = 0, []
    cams = {t["cam"] for t in pred} | {t["cam"] for t in gt}
    for cam in cams:
        P = [t for t in pred if t["cam"] == cam]
        G = [t for t in gt if t["cam"] == cam]
        if not P or not G:
            continue
        cost = np.array([[math.hypot(p["x"] - g["x"], p["y"] - g["y"]) for g in G] for p in P])
        big = cost.max() + tau + 1.0
        r, c = linear_sum_assignment(np.where(cost <= tau, cost, big))
        for i, j in zip(r, c):
            if cost[i, j] <= tau:
                tp += 1
                pairs.append((P[i], G[j], float(cost[i, j])))
    return {"tp": tp, "fp": len(pred) - tp, "fn": len(gt) - tp, "pairs": pairs}


def ground_tag(tag: dict, boxes_by_cam: Dict[str, List[dict]], halluc_dist: float = 100.0) -> dict:
    """grounded: inside some box of its camera; hallucinated: no box within ``halluc_dist`` px.
    ``metrics.evaluate`` additionally never counts a tag matched to a DriveLM GT tag as hallucinated."""
    boxes = boxes_by_cam.get(tag["cam"], [])
    inside = any(point_in_box(tag["x"], tag["y"], b["box2d"]) for b in boxes)
    nearest = min((dist_to_box(tag["x"], tag["y"], b["box2d"]) for b in boxes), default=float("inf"))
    return {"grounded": inside, "hallucinated": (not inside) and nearest > halluc_dist,
            "nearest_px": None if math.isinf(nearest) else round(nearest, 1)}
