"""Link DriveLM object tags to projected nuScenes boxes.

Rule: among boxes in the tag's camera that contain the tag point, pick the one
whose projected 3D centre (or 2D box centre) is closest to the point; ties go to
the smaller box, which is usually the nearer, unoccluded object. If no box
contains the point, fall back to the closest box centre within ``max_dist`` px.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

from .templates import Tag


def _center(b: dict):
    if b.get("center") is not None:
        return b["center"]
    x1, y1, x2, y2 = b["box2d"]
    return [(x1 + x2) / 2, (y1 + y2) / 2]


def point_in_box(x: float, y: float, box2d, margin: float = 0.0) -> bool:
    x1, y1, x2, y2 = box2d
    return x1 - margin <= x <= x2 + margin and y1 - margin <= y <= y2 + margin


def dist_to_box(x: float, y: float, box2d) -> float:
    x1, y1, x2, y2 = box2d
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return math.hypot(dx, dy)


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def link_tag(tag: Tag, boxes_by_cam: Dict[str, List[dict]], max_dist: float = 50.0,
             bbox2d: Optional[List[float]] = None, min_iou: float = 0.3) -> Optional[dict]:
    """Link one tag. When DriveLM provides the object's 2D box, IoU decides (robust to
    overlapping boxes); otherwise the point rule below is used."""
    boxes = boxes_by_cam.get(tag.cam, [])
    if bbox2d is not None and len(bbox2d) == 4 and boxes:
        best = max(boxes, key=lambda b: iou(bbox2d, b["box2d"]))
        score = iou(bbox2d, best["box2d"])
        if score >= min_iou:
            return {**best, "link": "iou", "link_dist": 0.0, "link_iou": round(score, 3)}
    inside = [b for b in boxes if point_in_box(tag.x, tag.y, b["box2d"])]
    if inside:
        def key(b):
            # distance to the visible (clipped) box centre: the projected 3D centre can lie far
            # outside the image for truncated objects; smaller (usually nearer) boxes win ties
            x1, y1, x2, y2 = b["box2d"]
            return (math.hypot((x1 + x2) / 2 - tag.x, (y1 + y2) / 2 - tag.y), (x2 - x1) * (y2 - y1))
        best = min(inside, key=key)
        return {**best, "link": "inside", "link_dist": 0.0}
    near = [(dist_to_box(tag.x, tag.y, b["box2d"]), b) for b in boxes]
    near = [(d, b) for d, b in near if d <= max_dist]
    if near:
        d, best = min(near, key=lambda t: t[0])
        return {**best, "link": "near", "link_dist": round(d, 1)}
    return None
