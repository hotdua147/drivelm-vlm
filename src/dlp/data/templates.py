"""Question templates, object tags and answer normalisation for DriveLM perception QA.

DriveLM refers to objects with tags of the form ``<c1,CAM_FRONT,1023.3,511.7>``:
an id, the camera whose image contains the object's centre, and the pixel
coordinates of that centre on the original 1600x900 image (origin top-left).

Question types (confirmed on DriveLM-nuScenes v1.1, 30 perception templates):

* P1  important objects                      -> list of tags                 (4,072 Q)
* P2  "moving status of object <tag>"         -> going ahead / turn left / ...  (8,777)
* P2o "observed status of object <tag>"       -> moving / stationary            (15,383)
* P3  "visual description of <tag>"           -> scored by object category      (17,282)
* P4  "Are there <attr> <cat> to the <region>" / "Is there any traffic element"
                                              -> yes / no                       (~41,000)
* P5  "What are objects to the <region>"      -> set of object categories       (19,985)
* P6  "What is the status of the <cat> that is/are to the <region>"
                                              -> moving / stationary            (37,828)
* other (scene description, relative positioning, lanes, traffic elements)
                                              -> trained on, never scored

Regions map one-to-one to cameras ("front left" -> CAM_FRONT_LEFT), which lets
P4 answers be checked against projected nuScenes boxes (see label_check).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

CAMERAS = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
]
IMG_W, IMG_H = 1600, 900

_NUM = r"-?\d+(?:\.\d+)?"
TAG_RE = re.compile(rf"<\s*(c\d+)\s*,\s*(CAM_[A-Z_]+)\s*,\s*({_NUM})\s*,\s*({_NUM})\s*>")


@dataclass(frozen=True)
class Tag:
    cid: str
    cam: str
    x: float
    y: float

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:  # canonical DriveLM spelling
        return format_tag(self)


def format_tag(tag: Tag) -> str:
    return f"<{tag.cid},{tag.cam},{tag.x:.1f},{tag.y:.1f}>"


def parse_tags(text: str) -> List[Tag]:
    """Return every well-formed tag in ``text``, in order, de-duplicated."""
    seen, out = set(), []
    for m in TAG_RE.finditer(text or ""):
        tag = Tag(m.group(1), m.group(2), float(m.group(3)), float(m.group(4)))
        key = (tag.cam, round(tag.x, 1), round(tag.y, 1))
        if key not in seen:
            seen.add(key)
            out.append(tag)
    return out


def count_malformed_tags(text: str) -> int:
    """Angle-bracket chunks that look like tags but do not parse (for parse-fail rate)."""
    candidates = re.findall(r"<[^<>]{0,60}>", text or "")
    return sum(1 for c in candidates if c.lstrip("<").strip().startswith("c") and not TAG_RE.fullmatch(c))


def tag_valid(tag: Tag) -> bool:
    return tag.cam in CAMERAS and 0 <= tag.x <= IMG_W and 0 <= tag.y <= IMG_H


# --------------------------------------------------------------------------- question types
_RULES = [
    ("P1", re.compile(r"^what are the important objects in the current scene", re.I)),
    ("P2", re.compile(r"moving status of object", re.I)),
    ("P2o", re.compile(r"observed status of object", re.I)),
    ("P3", re.compile(r"visual description of", re.I)),
    ("P6", re.compile(r"^what is the status of the .+ that (is|are) to the", re.I)),
    ("P5", re.compile(r"^what are objects to the", re.I)),
    ("P4", re.compile(r"^(are there|is there)\b", re.I)),
]
SCORED_QTYPES = ("P1", "P2", "P2o", "P3", "P4", "P5", "P6")


def classify_question(q: str) -> str:
    q = (q or "").strip()
    for name, rx in _RULES:
        if rx.search(q):
            return name
    return "other"


REGION_TO_CAM = {"front": "CAM_FRONT", "front left": "CAM_FRONT_LEFT", "front right": "CAM_FRONT_RIGHT",
                 "back": "CAM_BACK", "back left": "CAM_BACK_LEFT", "back right": "CAM_BACK_RIGHT"}
_REGION_RE = re.compile(r"to the (front|back)( left| right)? of the ego", re.I)


def region_camera(q: str):
    """Camera named by a question's region ("front view" counts as CAM_FRONT)."""
    m = _REGION_RE.search(q or "")
    if m:
        return REGION_TO_CAM[(m.group(1) + (m.group(2) or "")).lower()]
    if re.search(r"front view", q or "", re.I):
        return "CAM_FRONT"
    return None


_P4_RE = re.compile(r"^are there (?P<attr>parked|moving|standing)?\s*(?P<cat>[a-z ]+?)(?P<norider> without riders)? to the", re.I)
# P4 asks about one nuScenes class at a time ("parked trucks", "parked trailers" and "parked
# construction vehicles" are separate templates), so verification uses the exact class prefix.
P4_NUSC_CLASS = [
    (r"construction vehicles?", "vehicle.construction"), (r"trailers?", "vehicle.trailer"),
    (r"trucks?", "vehicle.truck"), (r"bus(es)?", "vehicle.bus"), (r"cars?", "vehicle.car"),
    (r"motorcycles?", "vehicle.motorcycle"), (r"bicycles?", "vehicle.bicycle"),
    (r"pedestrians?", "human.pedestrian"), (r"barriers?", "movable_object.barrier"),
    (r"traffic cones?", "movable_object.trafficcone"),
]


def parse_p4(q: str):
    """'Are there parked cars to the front of the ego car?' -> attr, group, nuScenes class, camera."""
    if re.search(r"traffic element", q or "", re.I):
        return {"attr": None, "group": "traffic_element", "nusc": None, "cam": "CAM_FRONT"}
    m = _P4_RE.search(q or "")
    if not m:
        return None
    attr = (m.group("attr") or "").lower() or ("without_rider" if m.group("norider") else None)
    cat = m.group("cat").strip().lower()
    nusc = next((cls for pat, cls in P4_NUSC_CLASS if re.fullmatch(pat, cat)), None)
    return {"attr": attr, "group": category_from_text(cat), "nusc": nusc, "cam": region_camera(q)}


# --------------------------------------------------------------------------- multiple choice
_OPT_RE = re.compile(r"(?:^|\s)([A-D])\.\s*(.+?)(?=\s+[A-D]\.\s|$)", re.S)


def parse_options(question: str) -> Dict[str, str]:
    """Options of a multiple-choice question: {'A': 'Going ahead.', ...}."""
    part = question.split("options:", 1)[1] if "options:" in question else question
    return {k: v.strip() for k, v in _OPT_RE.findall(part)}


def resolve_choice(answer: str, options: Dict[str, str]) -> str:
    """Map 'A' / 'A.' / 'A. Going ahead.' to the option text when options exist."""
    a = (answer or "").strip()
    m = re.match(r"^([A-D])(\.|\)|$)", a)
    if m and options and m.group(1) in options:
        return options[m.group(1)]
    return a


# --------------------------------------------------------------------------- moving status
STATUS_CLASSES = ["going_ahead", "turn_left", "turn_right", "stopped", "reversing", "other"]
_STATUS_PATTERNS = [
    ("turn_left", r"turn(ing)?\s+left|left turn"),
    ("turn_right", r"turn(ing)?\s+right|right turn"),
    ("reversing", r"revers|back(ing)? up"),
    # 'standing' found missing on 2026-11-10 (DriveLM: "One of the pedestrians is standing, and three are moving")
    ("stopped", r"stop|stationary|not moving|parked|\bstanding\b|\bsitting\b|\bwaiting\b|\bidle\b|static"),
    ("going_ahead", r"going ahead|go(ing)? straight|moving forward|driving forward|moving|ahead"),
]


def normalize_status(text: str) -> str:
    t = (text or "").lower()
    for label, pat in _STATUS_PATTERNS:
        if re.search(pat, t):
            return label
    return "other"


def status_from_nuscenes(attribute: Optional[str], speed: Optional[float]) -> str:
    """Coarse moving/stopped status implied by a nuScenes attribute (speed as fallback).

    Measured on DriveLM v1.1 (diag 2026-11-10): DriveLM labels ``vehicle.stopped`` objects
    'Stationary' (1,813 vs 282 'Moving'); counting stopped as moving drops agreement from
    89.7% to 79.8%, so stopped = stopped. ``cycle.with_rider`` says nothing about motion, so
    speed decides. nuScenes cannot tell 'turn left' from 'going ahead'.
    """
    a = (attribute or "").lower()
    if a.endswith(("parked", "stopped", "standing", "sitting_lying_down", "without_rider")):
        return "stopped"
    if a.endswith("moving"):
        return "moving"
    if speed is not None:  # with_rider, or no attribute
        return "moving" if speed > 0.5 else "stopped"
    return "unknown"


def coarse_status(label: str) -> str:
    return {"stopped": "stopped", "other": "unknown"}.get(label, "moving")


# Bicycles/motorcycles: DriveLM asks about the rider, not about motion (1,993 train answers such as
# "One motorcycle is without a rider." found on 2026-11-10). Separate labels, never folded into
# moving/stopped (a ridden bicycle can be waiting). "without" is checked before "with".
_NO_RIDER = re.compile(r"without (a )?riders?|(does not|doesn't|do not|don't) have (a )?riders?|\bno riders?\b|unridden")
_RIDER = re.compile(r"with (a )?riders?|(has|have) (a )?riders?|being ridden|\bridden\b")


def status_set(text: str) -> Optional[str]:
    """Statuses in an answer: 'Many trucks are parked.' -> 'stopped';
    'One car is moving, two cars are parked.' -> 'moving+stopped';
    'One of the bicycles has a rider, and one does not have a rider.' -> 'no_rider+rider'."""
    found = set()
    for clause in re.split(r"[,.;]| and ", (text or "").lower()):
        if _NO_RIDER.search(clause):
            found.add("no_rider")
            continue
        if _RIDER.search(clause):
            found.add("rider")
            continue
        lab = normalize_status(clause)
        if lab != "other":
            found.add(coarse_status(lab))
    return "+".join(sorted(found)) if found else None


# --------------------------------------------------------------------------- object category
# keyword -> nuScenes super-category used for P3 scoring and linking checks
# Order matters (first match wins in category_from_text). Riders come before pedestrians
# ("Person riding a motorcycle" is a motorcycle in nuScenes). nuScenes vehicle.car includes vans,
# minivans, SUVs, jeeps and wagons; vehicle.truck includes pickups. A bare "vehicle" names no
# category: it maps to "vehicle" (any vehicle), which never matches a specific nuScenes group.
CATEGORY_KEYWORDS = [
    ("motorcycle", r"motorcycles?|motorbikes?|scooters?|mopeds?"),
    ("bicycle", r"bicycles?|cyclists?|\bbikes?\b"),
    ("pedestrian", r"pedestrians?|person|people|\bman\b|\bwoman\b|child(ren)?|workers?|police officers?"),
    ("bus", r"\bbus(es)?\b"),
    ("truck", r"trucks?|lorr(y|ies)|pickups?|trailers?|construction vehicles?|tankers?"),
    ("car", r"\bcars?\b|sedans?|suvs?|hatchbacks?|taxis?|coupes?|\b(mini)?vans?\b|jeeps?|wagons?"),
    ("barrier", r"barriers?|fences?|bollards?"),
    ("traffic_cone", r"cones?"),
    # any sign counts ("Road sign." was missed in the 2026-11-11 hand check); pedestrians come first, so
    # "a person holding a sign" stays a pedestrian
    ("traffic_element", r"traffic (lights?|signs?|elements?)|signals?|\bsigns?\b|signposts?|signboards?"),
    ("vehicle", r"\bvehicles?\b"),
]
_EGO_RE = re.compile(r"\bego (car|vehicle)\b", re.I)

NUSC_TO_GROUP = {
    "human.pedestrian": "pedestrian",
    "vehicle.bicycle": "bicycle",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bus": "bus",
    "vehicle.truck": "truck",
    "vehicle.trailer": "truck",
    "vehicle.construction": "truck",
    "vehicle.emergency": "car",
    "vehicle.car": "car",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
}


def category_from_text(text: str) -> Optional[str]:
    """First matching category (priority order above); mentions of the ego car are ignored."""
    t = _EGO_RE.sub(" ", (text or "").lower())
    for group, pat in CATEGORY_KEYWORDS:
        if re.search(pat, t):
            return group
    return None


def categories_from_text(text: str) -> set:
    """All categories mentioned (for P5). 'construction vehicle' counts as truck, not car."""
    t = _EGO_RE.sub(" ", (text or "").lower())
    found = set()
    for group, pat in CATEGORY_KEYWORDS:
        if re.search(pat, t):
            found.add(group)
            t = re.sub(pat, " ", t)  # a span counts once, e.g. 'construction vehicle'
    return found


def category_group(nusc_category: str) -> str:
    for prefix, group in NUSC_TO_GROUP.items():
        if (nusc_category or "").startswith(prefix):
            return group
    return "other"


# --------------------------------------------------------------------------- yes / no
def normalize_yesno(text: str) -> Optional[str]:
    t = (text or "").strip().lower()
    if re.match(r"^yes\b", t):
        return "yes"
    if re.match(r"^(no|none|neither)\b", t):
        return "no"
    return None
