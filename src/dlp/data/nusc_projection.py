"""Minimal nuScenes metadata reader + 3D->2D box projection (numpy only).

Only the JSON tables are needed (no lidar/radar blobs). The maths follows
``nuscenes-devkit`` (``Box.corners`` + ``view_points``): a box in the global
frame is moved to the ego frame at the camera timestamp, then to the camera
frame, projected with the intrinsic matrix, and the 2D box is the extent of the
corners in front of the camera, clipped to the image. ``tests/test_projection.py``
checks the conventions on hand-built cases; on real data compare a few frames
with ``nuscenes.utils.geometry_utils.view_points`` (day-1 task).
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from .templates import CAMERAS, IMG_H, IMG_W, category_group

TABLES = [
    "scene", "sample", "sample_data", "calibrated_sensor", "sensor", "ego_pose",
    "sample_annotation", "instance", "category", "attribute", "log",
]


# --------------------------------------------------------------------------- geometry
def quat_to_rot(q) -> np.ndarray:
    """nuScenes quaternions are [w, x, y, z]."""
    w, x, y, z = (float(v) for v in q)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def box_corners(center, size_wlh, rotation_q) -> np.ndarray:
    """8 corners (3x8) of a box; nuScenes size is [width, length, height], x = length axis."""
    w, l, h = size_wlh
    x = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    corners = quat_to_rot(rotation_q) @ np.vstack([x, y, z])
    return corners + np.asarray(center, dtype=float).reshape(3, 1)


def global_to_camera(points: np.ndarray, ego_pose: dict, calib: dict) -> np.ndarray:
    """points 3xN global -> camera frame (x right, y down, z forward)."""
    p = points - np.asarray(ego_pose["translation"], dtype=float).reshape(3, 1)
    p = quat_to_rot(ego_pose["rotation"]).T @ p
    p = p - np.asarray(calib["translation"], dtype=float).reshape(3, 1)
    return quat_to_rot(calib["rotation"]).T @ p


def project_box(corners_cam: np.ndarray, intrinsic, min_depth: float = 0.1,
                img_w: int = IMG_W, img_h: int = IMG_H) -> Optional[List[float]]:
    """2D box [x1, y1, x2, y2] of the visible part, or None if not in view."""
    in_front = corners_cam[2] > min_depth
    if not in_front.any():
        return None
    K = np.asarray(intrinsic, dtype=float)
    pts = corners_cam[:, in_front]
    uv = (K @ pts)[:2] / pts[2]
    x1, y1 = uv.min(axis=1)
    x2, y2 = uv.max(axis=1)
    x1, x2 = np.clip([x1, x2], 0, img_w)
    y1, y2 = np.clip([y1, y2], 0, img_h)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return [float(x1), float(y1), float(x2), float(y2)]


def project_point(point_global, ego_pose, calib) -> Optional[List[float]]:
    p = global_to_camera(np.asarray(point_global, dtype=float).reshape(3, 1), ego_pose, calib)
    if p[2, 0] <= 0.1:
        return None
    uv = (np.asarray(calib["camera_intrinsic"], dtype=float) @ p)[:2, 0] / p[2, 0]
    return [float(uv[0]), float(uv[1])]


# --------------------------------------------------------------------------- metadata
class NuScenesLite:
    """Loads the JSON tables of ``v1.0-trainval`` (or ``v1.0-mini``) and indexes them."""

    def __init__(self, dataroot: str, version: str = "v1.0-trainval"):
        self.dataroot = dataroot
        folder = os.path.join(dataroot, version)
        self.t: Dict[str, Dict[str, dict]] = {}
        keep = {  # trainval sample_data / sample_annotation are large: keep only what is used
            "sample_data": ("token", "sample_token", "ego_pose_token", "calibrated_sensor_token", "filename", "is_key_frame"),
            "sample_annotation": ("token", "sample_token", "instance_token", "translation", "size", "rotation",
                                  "attribute_tokens", "prev", "next", "visibility_token"),
        }
        for name in TABLES:
            path = os.path.join(folder, f"{name}.json")
            with open(path) as f:
                rows = json.load(f)
            if name == "sample_data":  # sweeps are ~90% of the table and never used
                rows = [r for r in rows if r.get("is_key_frame", True)]
            if name in keep:
                rows = [{k: r[k] for k in keep[name] if k in r} for r in rows]
            self.t[name] = {r["token"]: r for r in rows}
            del rows
        used_poses = {sd["ego_pose_token"] for sd in self.t["sample_data"].values()}
        self.t["ego_pose"] = {k: v for k, v in self.t["ego_pose"].items() if k in used_poses}
        self._anns_by_sample = defaultdict(list)
        for ann in self.t["sample_annotation"].values():
            self._anns_by_sample[ann["sample_token"]].append(ann)
        self._sd_by_sample = defaultdict(dict)
        for sd in self.t["sample_data"].values():
            if not sd.get("is_key_frame", True):
                continue
            calib = self.t["calibrated_sensor"][sd["calibrated_sensor_token"]]
            channel = self.t["sensor"][calib["sensor_token"]]["channel"]
            self._sd_by_sample[sd["sample_token"]][channel] = sd

    # -- lookups -------------------------------------------------------------
    def get(self, table: str, token: str) -> dict:
        return self.t[table][token]

    def has_sample(self, token: str) -> bool:
        return token in self.t["sample"]

    def scene_of(self, sample_token: str) -> dict:
        return self.t["scene"][self.t["sample"][sample_token]["scene_token"]]

    def location_of(self, sample_token: str) -> str:
        return self.t["log"][self.scene_of(sample_token)["log_token"]]["location"]

    def category_name(self, ann: dict) -> str:
        inst = self.t["instance"][ann["instance_token"]]
        return self.t["category"][inst["category_token"]]["name"]

    def attribute_name(self, ann: dict) -> Optional[str]:
        toks = ann.get("attribute_tokens") or []
        return self.t["attribute"][toks[0]]["name"] if toks else None

    def ann_speed(self, ann: dict) -> Optional[float]:
        """Centre-difference speed (m/s) from prev/next annotations, as in the devkit."""
        prev_t, next_t = ann.get("prev") or "", ann.get("next") or ""
        a = self.t["sample_annotation"].get(prev_t, ann)
        b = self.t["sample_annotation"].get(next_t, ann)
        if a is b:
            return None
        ta = self.t["sample"][a["sample_token"]]["timestamp"]
        tb = self.t["sample"][b["sample_token"]]["timestamp"]
        dt = (tb - ta) / 1e6
        if dt <= 0 or dt > 1.5:
            return None
        d = np.asarray(b["translation"][:2]) - np.asarray(a["translation"][:2])
        return float(np.linalg.norm(d) / dt)

    # -- main entry ----------------------------------------------------------
    def gt_boxes(self, sample_token: str, cameras=CAMERAS, min_visible_px: float = 4.0) -> Dict[str, List[dict]]:
        """All annotations of a keyframe projected into each camera."""
        out = {cam: [] for cam in cameras}
        sds = self._sd_by_sample.get(sample_token, {})
        anns = self._anns_by_sample.get(sample_token, [])
        for cam in cameras:
            sd = sds.get(cam)
            if sd is None:
                continue
            calib = self.t["calibrated_sensor"][sd["calibrated_sensor_token"]]
            pose = self.t["ego_pose"][sd["ego_pose_token"]]
            for ann in anns:
                corners = box_corners(ann["translation"], ann["size"], ann["rotation"])
                corners_cam = global_to_camera(corners, pose, calib)
                box = project_box(corners_cam, calib["camera_intrinsic"])
                if box is None or min(box[2] - box[0], box[3] - box[1]) < min_visible_px:
                    continue
                cat = self.category_name(ann)
                out[cam].append({
                    "ann_token": ann["token"],
                    "category": cat,
                    "group": category_group(cat),
                    "attribute": self.attribute_name(ann),
                    "speed": self.ann_speed(ann),
                    "box2d": [round(v, 1) for v in box],
                    "center": project_point(ann["translation"], pose, calib),
                    # nuScenes visibility: 1 = 0-40%, 2 = 40-60%, 3 = 60-80%, 4 = 80-100% visible
                    "visibility": int(ann["visibility_token"]) if str(ann.get("visibility_token", "")).isdigit() else None,
                    "depth": round(float(corners_cam[2].mean()), 1),  # metres along the camera axis
                })
        return out

    def image_path(self, sample_token: str, cam: str) -> Optional[str]:
        sd = self._sd_by_sample.get(sample_token, {}).get(cam)
        return sd["filename"] if sd else None
