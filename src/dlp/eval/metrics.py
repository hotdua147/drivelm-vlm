"""All scored metrics, computed from ``_preds.jsonl`` + ``gt_boxes``.

A prediction record is the dataset row plus ``prediction`` (generated text) and,
optionally, ``latency_s``, ``n_image_tokens``, ``n_new_tokens``, ``peak_vram_gb``.
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional

import numpy as np

from ..data.templates import (
    STATUS_CLASSES, categories_from_text, category_from_text, count_malformed_tags, normalize_status,
    normalize_yesno, parse_options, parse_tags, region_camera, resolve_choice, status_set,
)
from .matching import ground_tag, match_tags

TAUS = (50, 100, 200)
COLLAPSE_MIN_SAMPLES = 20
COLLAPSE_MAX_DISTINCT = 2


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(100 * p, 2), round(100 * r, 2), round(100 * f, 2)


def macro_f1(y_true: List[str], y_pred: List[str], classes: Optional[Iterable[str]] = None) -> float:
    classes = sorted(set(classes or []) | set(y_true)) if classes else sorted(set(y_true))
    f1s = []
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        if tp + fp + fn == 0:
            continue
        f1s.append(_prf(tp, fp, fn)[2])
    return round(float(np.mean(f1s)), 2) if f1s else 0.0


def balanced_accuracy(y_true: List[str], y_pred: List[str]) -> float:
    recalls = []
    for c in sorted(set(y_true)):
        idx = [i for i, t in enumerate(y_true) if t == c]
        recalls.append(sum(1 for i in idx if y_pred[i] == c) / len(idx))
    return round(100 * float(np.mean(recalls)), 2) if recalls else 0.0


def _acc(y_true, y_pred):
    return round(100 * sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true), 2) if y_true else None


# ---------------------------------------------------------------- bootstrap CIs
BOOT_N = 1000
BOOT_SEED = 0


def _f1_from(tp, fp, fn):
    den = 2 * tp + fp + fn
    return 100 * 2 * tp / den if den else 0.0


def stat(kind: str, data: list, classes: Optional[Iterable[str]] = None) -> Optional[float]:
    """Headline statistic from per-row units.
    micro: rows of (tp, fp, fn) -> micro-F1 · ratio: rows of (num, den) -> 100*sum/sum ·
    acc / bacc / macro: rows of (y_true, y_pred) -> accuracy / balanced accuracy / macro-F1."""
    if not data:
        return None
    if kind == "micro":
        a = np.asarray(data, dtype=float).sum(axis=0)
        return _f1_from(*a)
    if kind == "ratio":
        a = np.asarray(data, dtype=float).sum(axis=0)
        return 100 * a[0] / a[1] if a[1] else 0.0
    yt, yp = [t for t, _ in data], [q for _, q in data]
    if kind == "acc":
        return 100 * sum(t == q for t, q in data) / len(data)
    if kind == "bacc":
        return balanced_accuracy(yt, yp)
    return macro_f1(yt, yp, classes)


def bootstrap_ci(kind: str, data: list, clusters: Optional[list] = None, classes=None,
                 B: int = BOOT_N, seed: int = BOOT_SEED) -> Optional[List[float]]:
    """95% percentile-bootstrap CI. With ``clusters`` (e.g. the scene of each row) whole clusters
    are resampled, so correlated questions from one scene stay together (wider, more honest CI)."""
    n = len(data)
    if n < 2:
        return None
    rng = np.random.default_rng(seed)
    if clusters is None:
        groups = [np.array([i]) for i in range(n)]
    else:
        by: Dict[object, list] = defaultdict(list)
        for i, c in enumerate(clusters):
            by[c].append(i)
        groups = [np.array(v) for v in by.values()]
        if len(groups) < 2:
            return None
    vals = []
    for _ in range(B):
        pick = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[k] for k in pick])
        v = stat(kind, [data[i] for i in idx], classes)
        if v is not None:
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(float(lo), 2), round(float(hi), 2)]


def paired_bootstrap(a_units: tuple, b_units: tuple, B: int = BOOT_N, seed: int = BOOT_SEED) -> Optional[dict]:
    """Difference a - b of one headline metric on the questions both runs answered, with a 95% CI from
    resampling whole scenes. Both runs see the same resampled questions, so val-set noise largely
    cancels: much tighter than comparing two separate CIs."""
    kind, a_rows, classes = a_units
    b = {i: d for i, d, _ in b_units[1]}
    rows = [(d, b[i], sc) for i, d, sc in a_rows if i in b]
    if len(rows) < 2:
        return None
    da, db = [x[0] for x in rows], [x[1] for x in rows]
    by: Dict[object, list] = defaultdict(list)
    for k, (_, _, sc) in enumerate(rows):
        by[sc].append(k)
    groups = [np.array(v) for v in by.values()]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(B):
        idx = np.concatenate([groups[k] for k in rng.integers(0, len(groups), size=len(groups))])
        va, vb = stat(kind, [da[i] for i in idx], classes), stat(kind, [db[i] for i in idx], classes)
        if va is not None and vb is not None:
            diffs.append(va - vb)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    point = stat(kind, da, classes) - stat(kind, db, classes)
    return {"diff": round(point, 2), "ci95": [round(float(lo), 2), round(float(hi), 2)], "n": len(rows),
            "p_le_0": round(float(np.mean(np.array(diffs) <= 0)), 4)}


def question_camera(r: dict) -> Optional[str]:
    """Camera a question is about: the camera of the tag in the question (P2/P2o/P3) or the
    camera named by the region (P4/P5/P6)."""
    q = [t for t in r.get("tags", []) if t.get("in") == "question"]
    if q:
        return q[0].get("cam")
    return region_camera(r.get("question", ""))


def load_gt_boxes(gt_dir: str, sample_token: str, cache: dict) -> Dict[str, List[dict]]:
    if sample_token not in cache:
        path = os.path.join(gt_dir, f"{sample_token}.json")
        cache[sample_token] = json.load(open(path)) if os.path.exists(path) else {}
    return cache[sample_token]


def _canon(qtype: str, r: dict) -> str:
    """Answer identity used for the collapse check (labels for closed-set types)."""
    p = r["prediction"]
    if qtype == "P2":
        return normalize_status(resolve_choice(p, parse_options(r["question"])))
    if qtype == "P3":
        return str(category_from_text(p))
    if qtype == "P4":
        return str(normalize_yesno(resolve_choice(p, parse_options(r["question"]))))
    if qtype in ("P2o", "P6"):
        return str(status_set(p))
    if qtype == "P5":
        return "|".join(sorted(categories_from_text(p)))
    if qtype == "P1":  # same set of tags = same answer
        return "|".join(sorted(f"{t.cam}:{round(t.x, -1)}:{round(t.y, -1)}" for t in parse_tags(p)))
    return p.strip().lower()


def _gt_tags(r: dict) -> List[dict]:
    return [t for t in r.get("tags", []) if t.get("in") == "answer"]


def evaluate_subset(rows: List[dict], qtype: str) -> dict:
    """Accuracy-type score of one closed-set question type on a subset of rows."""
    rows = [r for r in rows if r.get("answer_label")]
    if qtype == "P2":
        yt = [r["answer_label"] for r in rows]
        yp = [normalize_status(resolve_choice(r["prediction"], parse_options(r["question"]))) for r in rows]
        return {"accuracy": _acc(yt, yp), "macro_f1": macro_f1(yt, yp, STATUS_CLASSES)}
    if qtype == "P2o":
        yt = [r["answer_label"] for r in rows]
        yp = [status_set(r["prediction"]) or "none" for r in rows]
        return {"accuracy": _acc(yt, yp), "macro_f1": macro_f1(yt, yp)}
    yt = [r["answer_label"] for r in rows]
    yp = [normalize_yesno(r["prediction"]) or "none" for r in rows]
    return {"accuracy": _acc(yt, yp), "balanced_accuracy": balanced_accuracy(yt, yp)}


def evaluate(preds: List[dict], gt_dir: str, taus=TAUS, halluc_dist: float = 100.0, return_units: bool = False) -> dict:
    cache: dict = {}
    by_q = defaultdict(list)
    for r in preds:
        by_q[r["qtype"]].append(r)

    report: dict = {"n": len(preds), "n_by_qtype": {k: len(v) for k, v in by_q.items()}}
    # a model that answers nothing for a type (prior on P1, SigLIP on P4-P6) is "not applicable" there:
    # it is not scored and does not count towards the parse-fail rate
    unanswered = sorted(q for q, rs in by_q.items() if q != "other" and not any((r.get("prediction") or "").strip() for r in rs))
    for q in unanswered:
        del by_q[q]
    report["unanswered_qtypes"] = unanswered
    warnings: List[str] = []
    parse_fail = Counter()
    units: Dict[str, tuple] = {}  # headline metric -> (kind, per-row data, rows, classes) for CIs and splits

    # ---- P1: important objects + grounding / hallucination
    p1 = by_q.get("P1", [])
    if p1:
        res = {}
        for tau in taus:
            tp = fp = fn = 0
            for r in p1:
                m = match_tags([t.to_dict() for t in parse_tags(r["prediction"])], _gt_tags(r), tau)
                tp, fp, fn = tp + m["tp"], fp + m["fp"], fn + m["fn"]
            res[f"tau{tau}"] = dict(zip(("precision", "recall", "f1"), _prf(tp, fp, fn)))
        report["P1"] = res
        n_tags = n_ground = n_hall = 0
        cam_split = {"front": [0, 0, 0], "other": [0, 0, 0]}  # tp, fp, fn at tau=100
        p1_counts, hall_pairs = [], []
        for r in p1:
            pts = [t.to_dict() for t in parse_tags(r["prediction"])]
            m100 = match_tags(pts, _gt_tags(r), 100)
            p1_counts.append((m100["tp"], m100["fp"], m100["fn"]))
            h0 = n_hall
            if not pts and count_malformed_tags(r["prediction"]):
                parse_fail["P1"] += 1
            boxes = load_gt_boxes(gt_dir, r["sample_token"], cache)
            # a tag matched to a DriveLM ground-truth tag is never a hallucination, even when nuScenes
            # has no box there (traffic lights/signs are not annotated in nuScenes)
            matched = {id(p) for p, _, _ in match_tags(pts, _gt_tags(r), 100)["pairs"]}
            for t in pts:
                g = ground_tag(t, boxes, halluc_dist)
                n_tags += 1
                n_ground += g["grounded"] or id(t) in matched
                n_hall += g["hallucinated"] and id(t) not in matched
            hall_pairs.append((n_hall - h0, len(pts)))
            for key, sel in (("front", lambda t: t["cam"] == "CAM_FRONT"), ("other", lambda t: t["cam"] != "CAM_FRONT")):
                m = match_tags([t for t in pts if sel(t)], [t for t in _gt_tags(r) if sel(t)], 100)
                cam_split[key] = [a + b for a, b in zip(cam_split[key], (m["tp"], m["fp"], m["fn"]))]
        report["grounding_rate"] = round(100 * n_ground / n_tags, 2) if n_tags else None
        report["hallucination_rate"] = round(100 * n_hall / n_tags, 2) if n_tags else None
        report["n_pred_tags"] = n_tags
        report["P1_by_camera_tau100"] = {k: _prf(*v)[2] for k, v in cam_split.items()}
        units["P1_f1_tau100"] = ("micro", p1_counts, p1, None)
        if n_tags:
            units["hallucination_rate"] = ("ratio", hall_pairs, p1, None)
        report["pred_tags_per_answer"] = round(n_tags / len(p1), 2)

    # ---- P2: moving status
    p2 = by_q.get("P2", [])
    if p2:
        yt = [r["answer_label"] or normalize_status(r["answer"]) for r in p2]
        yp = [normalize_status(resolve_choice(r["prediction"], parse_options(r["question"]))) for r in p2]
        parse_fail["P2"] = sum(1 for p in yp if p == "other")
        turns = ("turn_left", "turn_right")
        gt_turn = [i for i, t in enumerate(yt) if t in turns]
        pr_turn = [i for i, p in enumerate(yp) if p in turns]
        report["P2"] = {
            "accuracy": _acc(yt, yp), "macro_f1": macro_f1(yt, yp, STATUS_CLASSES),
            "balanced_accuracy": balanced_accuracy(yt, yp),
            "n": len(yt), "n_turn": len(gt_turn), "label_counts": dict(Counter(yt)),
            # exact direction / any turn, among ground-truth turns; precision among predicted turns
            "turn_recall": round(100 * sum(yp[i] == yt[i] for i in gt_turn) / len(gt_turn), 2) if gt_turn else None,
            "turn_any_recall": round(100 * sum(yp[i] in turns for i in gt_turn) / len(gt_turn), 2) if gt_turn else None,
            "turn_precision": round(100 * sum(yt[i] == yp[i] for i in pr_turn) / len(pr_turn), 2) if pr_turn else None,
        }
        n_any = sum(yt[i] in turns for i in pr_turn)
        report["P2"]["turn_any_precision"] = round(100 * n_any / len(pr_turn), 2) if pr_turn else None
        report["P2"]["pred_turn_rate"] = round(100 * len(pr_turn) / len(yp), 2)
        units["P2_macro_f1"] = ("macro", list(zip(yt, yp)), p2, STATUS_CLASSES)
        units["P2_balanced_accuracy"] = ("bacc", list(zip(yt, yp)), p2, None)

    # ---- P3: object category of the DriveLM answer (what the model is trained on); the nuScenes
    #      class of the linked box is reported as a second, independent check
    p3 = by_q.get("P3", [])
    if p3:
        yt, yp, yn, p3_rows = [], [], [], []
        for r in p3:
            if r.get("answer_label") in (None, "vehicle"):
                continue
            p3_rows.append(r)
            pred = category_from_text(r["prediction"]) or "none"
            yt.append(r["answer_label"])
            yp.append(pred)
            gq = [t for t in r.get("tags", []) if t.get("in") == "question" and t.get("group")]
            if gq:
                yn.append((gq[0]["group"], pred))
        parse_fail["P3"] = sum(1 for p in yp if p == "none")
        report["P3"] = {"category_accuracy": _acc(yt, yp), "n_scored": len(yt),
                        "category_accuracy_vs_nuscenes": _acc([a for a, _ in yn], [b for _, b in yn])}
        units["P3_category_accuracy"] = ("acc", list(zip(yt, yp)), p3_rows, None)

    # ---- P4: yes / no
    p4 = by_q.get("P4", [])
    if p4:
        pairs = [(r["answer_label"], normalize_yesno(resolve_choice(r["prediction"], parse_options(r["question"]))))
                 for r in p4 if r.get("answer_label")]
        yt = [a for a, _ in pairs]
        yp = [b or "none" for _, b in pairs]
        parse_fail["P4"] = sum(1 for p in yp if p == "none")
        report["P4"] = {"accuracy": _acc(yt, yp), "balanced_accuracy": balanced_accuracy(yt, yp), "n_scored": len(yt)}
        units["P4_balanced_accuracy"] = ("bacc", list(zip(yt, yp)), [r for r in p4 if r.get("answer_label")], None)

    # ---- P2o / P6: coarse status (moving / stopped / both)
    for qt in ("P2o", "P6"):
        rs = by_q.get(qt, [])
        pairs = [(r["answer_label"], status_set(resolve_choice(r["prediction"], parse_options(r["question"]))))
                 for r in rs if r.get("answer_label")]
        if pairs:
            yt = [a for a, _ in pairs]
            yp = [b or "none" for _, b in pairs]
            parse_fail[qt] = sum(1 for p in yp if p == "none")
            report[qt] = {"accuracy": _acc(yt, yp), "macro_f1": macro_f1(yt, yp), "n_scored": len(yt)}
            units[f"{qt}_macro_f1"] = ("macro", list(zip(yt, yp)), [r for r in rs if r.get("answer_label")], None)

    # ---- P5: objects in a region -> micro P/R/F1 over the category sets
    p5 = by_q.get("P5", [])
    if p5:
        tp = fp = fn = 0
        p5_counts = []
        for r in p5:
            gt_set = set(filter(None, (r.get("answer_label") or "").split("|"))) - {"none"}
            pr_set = categories_from_text(r["prediction"])
            c = (len(gt_set & pr_set), len(pr_set - gt_set), len(gt_set - pr_set))
            p5_counts.append(c)
            tp, fp, fn = tp + c[0], fp + c[1], fn + c[2]
        report["P5"] = dict(zip(("precision", "recall", "f1"), _prf(tp, fp, fn)))
        units["P5_f1"] = ("micro", p5_counts, p5, None)

    # ---- label-noise-aware scores: drop rows whose DriveLM label contradicts nuScenes
    clean = {}
    for qt in ("P2", "P2o", "P4"):
        rs = [r for r in by_q.get(qt, []) if "nusc_disagree" not in (r.get("label_flags") or [])]
        n_all = len(by_q.get(qt, []))
        if not rs or len(rs) == n_all:
            continue
        sub = evaluate_subset(rs, qt)
        sub["n"], sub["dropped"] = len(rs), n_all - len(rs)
        clean[qt] = sub
    report["clean_subset"] = clean

    # ---- robustness: parse failures, distinct answers, collapse
    scored = sum(len(by_q.get(k, [])) for k in ("P1", "P2", "P2o", "P3", "P4", "P5", "P6"))
    report["parse_fail_rate"] = round(100 * sum(parse_fail.values()) / scored, 2) if scored else None
    distinct = {}
    for q, rs in by_q.items():
        answers = {_canon(q, r) for r in rs}
        d = len(answers)
        distinct[q] = d
        if q == "P1" and answers == {""}:
            continue  # model that does not answer P1 (prior / SigLIP baselines)
        # yes/no and moving/stopped questions have very few valid answers
        limit = 1 if q in ("P4", "P2o", "P6") else COLLAPSE_MAX_DISTINCT
        if q != "other" and len(rs) >= COLLAPSE_MIN_SAMPLES and d <= limit:
            warnings.append(f"COLLAPSE: {q} has {d} distinct answers over {len(rs)} samples")
    report["distinct_answers"] = distinct

    # ---- subsets (generalisation): P1 F1@100 and hallucination
    subsets = {}
    for key in ("city", "condition"):
        for val in sorted({r.get(key) for r in p1 if r.get(key)}):
            rs = [r for r in p1 if r.get(key) == val]
            tp = fp = fn = nt = nh = 0
            for r in rs:
                pts = [t.to_dict() for t in parse_tags(r["prediction"])]
                m = match_tags(pts, _gt_tags(r), 100)
                tp, fp, fn = tp + m["tp"], fp + m["fp"], fn + m["fn"]
                boxes = load_gt_boxes(gt_dir, r["sample_token"], cache)
                matched = {id(p) for p, _, _ in m["pairs"]}
                for t in pts:
                    nt += 1
                    nh += ground_tag(t, boxes, halluc_dist)["hallucinated"] and id(t) not in matched
            subsets[f"{key}={val}"] = {"n": len(rs), "P1_f1_tau100": _prf(tp, fp, fn)[2],
                                       "hallucination_rate": round(100 * nh / nt, 2) if nt else None}
    report["subsets"] = subsets

    # ---- cost
    # latency over answered questions only (an empty answer for an unsupported type takes ~0 s)
    lat = [r["latency_s"] for r in preds if r.get("latency_s") is not None and (r.get("prediction") or "").strip()]
    if lat:
        report["latency_s"] = {"p50": round(float(np.percentile(lat, 50)), 4),
                               "p95": round(float(np.percentile(lat, 95)), 4)}
    vram = [r["peak_vram_gb"] for r in preds if r.get("peak_vram_gb") is not None]
    if vram:
        report["peak_vram_gb"] = round(max(vram), 2)
    itok = [r["n_image_tokens"] for r in preds if r.get("n_image_tokens") is not None]
    if itok:
        report["image_tokens_mean"] = round(float(np.mean(itok)), 1)

    # ---- uncertainty: 95% bootstrap CIs, resampling whole scenes (questions of one scene are correlated)
    #      and, for comparison, single questions
    ci_scene, ci_q, by_cam = {}, {}, {}
    for key, (kind, data, rows, classes) in units.items():
        assert len(data) == len(rows), key
        c = bootstrap_ci(kind, data, [r.get("scene_token") or r.get("sample_token") for r in rows], classes)
        if c:
            ci_scene[key] = c
        c = bootstrap_ci(kind, data, None, classes)
        if c:
            ci_q[key] = c
        # ---- split by the camera the question is about (front vs the five others)
        if key.startswith(("P2o", "P3", "P4", "P5", "P6")):
            cams = [question_camera(r) for r in rows]
            f = [d for d, cam in zip(data, cams) if cam == "CAM_FRONT"]
            o = [d for d, cam in zip(data, cams) if cam and cam != "CAM_FRONT"]
            v = lambda x: round(stat(kind, x, classes), 2) if x else None
            by_cam[key] = {"front": v(f), "other": v(o), "n_front": len(f), "n_other": len(o)}
    report["ci95"] = ci_scene
    report["ci95_question"] = ci_q
    report["ci95_meta"] = {"method": "percentile bootstrap, resampling scenes (ci95) or questions (ci95_question)",
                           "B": BOOT_N, "seed": BOOT_SEED}
    report["by_question_camera"] = by_cam
    if return_units:  # per-question units for paired comparisons between runs (not written to disk)
        report["_units"] = {k: (kind, [(r["id"], d, r.get("scene_token") or r.get("sample_token")) for d, r in zip(data, rows)], classes)
                            for k, (kind, data, rows, classes) in units.items()}
    report["warnings"] = warnings
    return report
