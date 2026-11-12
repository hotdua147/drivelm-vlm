"""Reading ``v1_1_train_nus.json`` (DriveLM-nuScenes v1.1, Hugging Face ``OpenDriveLab/DriveLM``).

Layout (confirmed on the real file on 2026-11-10)::

    {scene_token: {"scene_description": str,
                   "key_frames": {frame_token: {"key_object_infos": {tag: {...}},
                                                "QA": {"perception": [...], "prediction": [...],
                                                       "planning": [...], "behavior": [...]},
                                                "image_paths": {CAM: path}}}}}

Each QA item has at least ``Q`` and ``A``; ``frame_token`` equals the nuScenes
``sample_token``. ``key_object_infos`` maps a tag string to fields such as
``Category``, ``Status``, ``Visual_description`` and ``2d_bbox`` (confirmed), e.g.
``'<c1,CAM_BACK,1088.3,497.5>': {'Category': 'Vehicle', 'Status': 'Moving',
'Visual_description': 'Brown SUV.', '2d_bbox': [966.6, 403.3, 1224.1, 591.7]}``.
QA items also carry ``C``, ``con_up``, ``con_down``, ``cluster``, ``layer`` (graph links, unused here).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

from .templates import Tag, parse_tags


@dataclass
class KeyObject:
    tag: Tag
    category: Optional[str] = None
    status: Optional[str] = None
    description: Optional[str] = None
    bbox2d: Optional[List[float]] = None


@dataclass
class Frame:
    scene_token: str
    sample_token: str
    scene_description: str
    image_paths: Dict[str, str]
    key_objects: List[KeyObject] = field(default_factory=list)
    qa: Dict[str, List[dict]] = field(default_factory=dict)


def _first(d: dict, *keys):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def load_drivelm(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def iter_frames(data: dict) -> Iterator[Frame]:
    for scene_token, scene in data.items():
        desc = scene.get("scene_description", "")
        for frame_token, fr in (scene.get("key_frames") or {}).items():
            objs = []
            for tag_str, info in (fr.get("key_object_infos") or {}).items():
                tags = parse_tags(tag_str)
                if not tags:
                    continue
                info = info or {}
                objs.append(KeyObject(
                    tag=tags[0],
                    category=_first(info, "Category", "category"),
                    status=_first(info, "Status", "status"),
                    description=_first(info, "Visual_description", "visual_description"),
                    bbox2d=_first(info, "2d_bbox", "bbox", "2d_box"),
                ))
            yield Frame(
                scene_token=scene_token,
                sample_token=frame_token,
                scene_description=desc,
                image_paths=dict(fr.get("image_paths") or {}),
                key_objects=objs,
                qa={k: list(v or []) for k, v in (fr.get("QA") or {}).items()},
            )


def iter_qa(frame: Frame, categories=("perception",)) -> Iterator[tuple]:
    for cat in categories:
        for i, item in enumerate(frame.qa.get(cat, [])):
            q, a = item.get("Q"), item.get("A")
            if q is None or a is None:
                continue
            yield cat, i, q, a
