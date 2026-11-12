"""Draw GT boxes, GT tags and predicted tags on the 6 cameras as one 3x2 mosaic."""
from __future__ import annotations

import os
from typing import Dict, List, Optional

from PIL import Image, ImageDraw

from ..data.dataset import GRID_ORDER

GT_BOX = (90, 160, 255)
GT_TAG = (40, 200, 90)
PRED_TAG = (235, 60, 60)


def _cross(d: ImageDraw.ImageDraw, x, y, r, color, width=3):
    d.line([(x - r, y - r), (x + r, y + r)], fill=color, width=width)
    d.line([(x - r, y + r), (x + r, y - r)], fill=color, width=width)


def draw_sample(image_root: str, images: Dict[str, str], gt_boxes: Dict[str, List[dict]],
                gt_tags: List[dict], pred_tags: List[dict], title: Optional[str] = None,
                cell=(640, 360)) -> Image.Image:
    w, h = cell
    sx, sy = w / 1600.0, h / 900.0
    top = 28 if title else 0
    canvas = Image.new("RGB", (3 * w, 2 * h + top), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    if title:
        d.text((8, 8), title, fill=(0, 0, 0))
    for r, row in enumerate(GRID_ORDER):
        for c, cam in enumerate(row):
            rel = images.get(cam)
            img = Image.open(os.path.join(image_root, rel)).convert("RGB").resize(cell) if rel and os.path.exists(os.path.join(image_root, rel)) else Image.new("RGB", cell, (30, 30, 30))
            dd = ImageDraw.Draw(img)
            for b in gt_boxes.get(cam, []):
                x1, y1, x2, y2 = b["box2d"]
                dd.rectangle([x1 * sx, y1 * sy, x2 * sx, y2 * sy], outline=GT_BOX, width=2)
            for t in gt_tags:
                if t["cam"] == cam:
                    dd.ellipse([t["x"] * sx - 7, t["y"] * sy - 7, t["x"] * sx + 7, t["y"] * sy + 7], outline=GT_TAG, width=3)
            for t in pred_tags:
                if t["cam"] == cam:
                    _cross(dd, t["x"] * sx, t["y"] * sy, 7, PRED_TAG)
            dd.text((6, 4), cam, fill=(255, 255, 0))
            canvas.paste(img, (c * w, top + r * h))
    return canvas
