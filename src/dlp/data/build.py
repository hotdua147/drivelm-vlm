"""raw DriveLM + nuScenes metadata -> data/processed (jsonl, gt_boxes, stats, label_check)."""
from __future__ import annotations

import json
import os
import random
from collections import Counter
from typing import Dict, List, Optional

from .drivelm_io import iter_frames, iter_qa, load_drivelm
from .label_check import LabelChecker
from .linking import link_tag
from .nusc_projection import NuScenesLite
from .split import city_of, condition_of, scene_split
from .templates import (
    CAMERAS, SCORED_QTYPES, categories_from_text, category_from_text, classify_question, coarse_status,
    format_tag, normalize_status, normalize_yesno, parse_options, parse_p4, parse_tags,
    resolve_choice, status_from_nuscenes, status_set, tag_valid,
)

SCORED = SCORED_QTYPES
# DriveLM key-object categories with no nuScenes annotation (1,403 of 1,875 traffic elements
# found no box on v1.1; the remaining 25% were links to poles/barriers, i.e. wrong)
UNVERIFIABLE_CATEGORIES = {"traffic element"}


def _norm_image_path(p: str) -> str:
    # DriveLM stores "../nuscenes/samples/CAM_FRONT/xxx.jpg"; keep it relative to the nuScenes root
    i = p.find("samples/")
    return p[i:] if i >= 0 else p


def _answer_label(qtype: str, question: str, answer: str) -> Optional[str]:
    a = resolve_choice(answer, parse_options(question))
    if qtype == "P2":
        return normalize_status(a)
    if qtype in ("P2o", "P6"):
        return status_set(a)
    if qtype == "P3":
        return category_from_text(a)
    if qtype == "P4":
        return normalize_yesno(a)
    if qtype == "P5":
        return "|".join(sorted(categories_from_text(a))) or "none"
    return None


CAM_RING = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"]
# P4 definition variants measured by the label check; the best-agreeing one is used for flags.
P4_VARIANTS = {
    "exact": {"parked_includes_stopped": False, "neighbours": False},
    "parked_incl_stopped": {"parked_includes_stopped": True, "neighbours": False},
    "with_neighbour_cams": {"parked_includes_stopped": False, "neighbours": True},
    "both": {"parked_includes_stopped": True, "neighbours": True},
}


def _attr_ok(attr: Optional[str], nusc_attr: Optional[str], parked_includes_stopped: bool = False) -> bool:
    if attr is None:
        return True
    a = (nusc_attr or "").lower()
    parked = ("parked", "stopped") if parked_includes_stopped else ("parked",)
    return {"parked": a.endswith(parked), "moving": a.endswith("moving"),
            "standing": a.endswith(("standing", "sitting_lying_down")),
            "without_rider": a.endswith("without_rider")}.get(attr, False)


def p4_from_nuscenes(question: str, gt: Dict[str, List[dict]], parked_includes_stopped: bool = False,
                     neighbours: bool = False) -> Optional[str]:
    """yes/no implied by projected nuScenes boxes of the asked class, or None when nuScenes cannot
    tell (traffic lights/signs are not annotated)."""
    p = parse_p4(question)
    if not p or p["nusc"] is None or p["cam"] is None:
        return None
    cams = [p["cam"]]
    if neighbours:
        i = CAM_RING.index(p["cam"])
        cams += [CAM_RING[(i - 1) % 6], CAM_RING[(i + 1) % 6]]
    hit = any(b["category"].startswith(p["nusc"]) and _attr_ok(p["attr"], b["attribute"], parked_includes_stopped)
              for c in cams for b in gt.get(c, []))
    return "yes" if hit else "no"


def build(drivelm_json: str, nusc_root: str, out_dir: str, version: str = "v1.0-trainval",
          categories=("perception",), cameras=CAMERAS, val_frac: float = 0.1, seed: int = 0,
          n_val_fixed: int = 500, thresholds: Optional[Dict[str, float]] = None,
          data_version: str = "d1") -> dict:
    os.makedirs(os.path.join(out_dir, "gt_boxes"), exist_ok=True)
    data = load_drivelm(drivelm_json)
    nusc = NuScenesLite(nusc_root, version)
    lc = LabelChecker(thresholds)
    rows: Dict[str, List[dict]] = {"train": [], "val": []}

    for fr in iter_frames(data):
        found = nusc.has_sample(fr.sample_token)
        lc.frame(found, fr.sample_token)
        if not found:
            continue
        gt = nusc.gt_boxes(fr.sample_token, cameras)
        with open(os.path.join(out_dir, "gt_boxes", f"{fr.sample_token}.json"), "w") as f:
            json.dump(gt, f)
        split = scene_split(fr.scene_token, val_frac, seed)
        city = city_of(nusc.location_of(fr.sample_token))
        cond = condition_of(nusc.scene_of(fr.sample_token).get("description", "") or fr.scene_description)
        images = {cam: _norm_image_path(fr.image_paths[cam]) if cam in fr.image_paths
                  else nusc.image_path(fr.sample_token, cam) for cam in cameras}

        # key objects: link + compare DriveLM fields with nuScenes labels
        key_links, unverifiable = {}, set()
        for ko in fr.key_objects:
            key = (ko.tag.cam, round(ko.tag.x, 1), round(ko.tag.y, 1))
            if (ko.category or "").strip().lower() in UNVERIFIABLE_CATEGORIES:
                # traffic lights/signs are not annotated in nuScenes: never link (a link would be wrong)
                lc.unverifiable(ko.category)
                key_links[key] = None
                unverifiable.add(key)
                continue
            link = link_tag(ko.tag, gt, bbox2d=ko.bbox2d)
            lc.tag(tag_valid(ko.tag), link, format_tag(ko.tag), fr.sample_token, ko.tag.cam, ko.category)
            key_links[key] = link
            if link is not None:
                ctx = {"sample_token": fr.sample_token, "tag": format_tag(ko.tag)}
                if ko.status:
                    spd = link["speed"]
                    lc.status(coarse_status(normalize_status(ko.status)),
                              status_from_nuscenes(link["attribute"], spd), ctx,
                              raw=(ko.status, link["attribute"] or "no_attr",
                                   "none" if spd is None else (">0.5" if spd > 0.5 else "<=0.5")))
                lc.category(category_from_text(f"{ko.category or ''} {ko.description or ''}"),
                            link["group"], {**ctx, "desc": ko.description, "nusc_category": link["category"]})

        for cat, i, q, a in iter_qa(fr, categories):
            qtype = classify_question(q)
            label = _answer_label(qtype, q, a)
            lc.question(qtype, q, label)
            nusc_check = None
            if qtype == "P4" and label is not None:
                nusc_check = {v: p4_from_nuscenes(q, gt, **kw) for v, kw in P4_VARIANTS.items()}
                lc.p4(label, nusc_check, {"sample_token": fr.sample_token, "q": q}, parse_p4(q))
            tags = []
            for t in parse_tags(q) + parse_tags(a):
                k = (t.cam, round(t.x, 1), round(t.y, 1))
                link = key_links[k] if k in key_links else link_tag(t, gt)
                tags.append({**t.to_dict(), "in": "question" if t in parse_tags(q) else "answer",
                             "ann_token": link["ann_token"] if link else None,
                             "category": link["category"] if link else None,
                             "group": link["group"] if link else None,
                             "attribute": link["attribute"] if link else None,
                             "speed": link["speed"] if link else None,
                             "box2d": link["box2d"] if link else None,
                             "verifiable": k not in unverifiable})
            flags = [f"{'unverifiable' if not t['verifiable'] else 'unlinked'}:{t['cid']}"
                     for t in tags if t["ann_token"] is None]
            if qtype in ("P2", "P2o") and label is not None:
                qtag = next((t for t in tags if t["in"] == "question" and t["ann_token"]), None)
                if qtag is not None:
                    ns = status_from_nuscenes(qtag["attribute"], qtag["speed"])
                    dl = coarse_status(label) if qtype == "P2" else label
                    if ns != "unknown" and dl in ("moving", "stopped") and ns != dl:
                        flags.append("nusc_disagree")
            rows[split].append({
                "id": f"{fr.sample_token}_{cat[:4]}_{i:02d}",
                "scene_token": fr.scene_token,
                "sample_token": fr.sample_token,
                "split": split, "city": city, "condition": cond,
                "images": images,
                "category": cat, "qtype": qtype,
                "question": q, "answer": a, "answer_label": label,
                "tags": tags,
                "label_flags": flags,
                "_p4_nusc": nusc_check,
            })

    summary = lc.summary()
    best = summary.get("p4_best_variant")
    for rs in rows.values():
        for r in rs:
            chk = r.pop("_p4_nusc", None)
            if best and chk and chk.get(best) is not None and chk[best] != r["answer_label"]:
                r["label_flags"].append("nusc_disagree")
    for split, rs in rows.items():
        with open(os.path.join(out_dir, f"{split}.jsonl"), "w") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # fixed val ids, stratified so that rare types (P1 is ~2.5% of perception) are well represented
    rng = random.Random(seed)
    by_type: Dict[str, List[str]] = {}
    for r in rows["val"]:
        if r["qtype"] in SCORED:
            by_type.setdefault(r["qtype"], []).append(r["id"])
    per_type = n_val_fixed // max(1, len(by_type))
    fixed = []
    for qt in sorted(by_type):
        ids = sorted(by_type[qt])
        fixed += rng.sample(ids, min(per_type, len(ids)))
    fixed = sorted(fixed)
    with open(os.path.join(out_dir, f"val_{n_val_fixed}_ids.txt"), "w") as f:
        f.write("\n".join(fixed) + ("\n" if fixed else ""))

    stats = {
        "data_version": data_version,
        "n_rows": {k: len(v) for k, v in rows.items()},
        "n_scenes": {k: len({r["scene_token"] for r in v}) for k, v in rows.items()},
        "qtype": {k: dict(Counter(r["qtype"] for r in v)) for k, v in rows.items()},
        "city": {k: dict(Counter(r["city"] for r in v)) for k, v in rows.items()},
        "condition": {k: dict(Counter(r["condition"] for r in v)) for k, v in rows.items()},
        "n_val_fixed": len(fixed),
    }
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(out_dir, "label_check.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "VERSION"), "w") as f:
        f.write(data_version + "\n")
    return {"stats": stats, "label_check": summary}
