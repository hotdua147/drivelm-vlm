"""Where do DriveLM's "turn" labels come from? (review question 20; CPU only, metadata only)

For every P2 question ("moving status of object <c..>") whose object is linked to a nuScenes
annotation, measure how much the object's heading (box yaw) changes in the PAST (t-h -> t) and in
the FUTURE (t -> t+h), for h = 1, 2, 3 s, using the annotation prev/next chain (keyframes, 2 Hz).
A threshold rule on |yaw change| (+ sign for left/right) is tuned on train and scored on val:
  * if the future window explains the labels much better than the past one, P2 is a prediction
    question -> multi-frame history can only help partly;
  * the best rule is an oracle: an upper bound for any model that sees the object's true motion.

    python scripts/label_oracle.py            # -> results/label_oracle.md
"""
import argparse
import math
import os
from collections import Counter

import numpy as np

from _common import rel  # noqa: F401  (adds src/ to sys.path)
from dlp.data.nusc_projection import NuScenesLite
from dlp.utils.config import load_config
from dlp.utils.runs import read_jsonl

LABELS = ("going_ahead", "turn_left", "turn_right")
HORIZONS = (1, 2, 3)  # seconds = 2, 4, 6 keyframe steps


def yaw(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def walk(nusc, ann, key, steps):
    for _ in range(steps):
        nxt = ann.get(key) or ""
        if nxt not in nusc.t["sample_annotation"]:
            return None
        ann = nusc.t["sample_annotation"][nxt]
    return ann


def yaw_change(nusc, ann, h, direction):
    """Signed heading change in degrees (+ = counter-clockwise seen from above = left turn)."""
    other = walk(nusc, ann, "prev" if direction == "past" else "next", 2 * h)
    if other is None:
        return None
    d = wrap(yaw(ann["rotation"]) - yaw(other["rotation"]))
    return math.degrees(d if direction == "past" else -d)


def predict(change, thr):
    if change is None:
        return None
    return "going_ahead" if abs(change) < thr else ("turn_left" if change > 0 else "turn_right")


def score(rows, key, thr):
    yt, yp = [], []
    for r in rows:
        p = predict(r[key], thr)
        if p is not None:
            yt.append(r["label"]), yp.append(p)
    if not yt:
        return None
    rec = [np.mean([p == c for t, p in zip(yt, yp) if t == c]) for c in LABELS if c in yt]
    turns = [(t, p) for t, p in zip(yt, yp) if t != "going_ahead"]
    return {"n": len(yt), "bal_acc": 100 * float(np.mean(rec)),
            "turn_recall": 100 * float(np.mean([t == p for t, p in turns])) if turns else None,
            "pred_turn_rate": 100 * float(np.mean([p != "going_ahead" for p in yp]))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/data/perception.yaml")
    ap.add_argument("--out", default="results/label_oracle.md")
    a = ap.parse_args()
    cfg = load_config(rel(a.config))
    P = cfg["paths"]
    nusc = NuScenesLite(rel(P["nusc_root"]), P["nusc_version"])
    data = {}
    for split in ("train", "val"):
        rows = []
        for r in read_jsonl(os.path.join(rel(P["processed"]), f"{split}.jsonl")):
            if r["qtype"] != "P2" or r.get("answer_label") not in LABELS:
                continue
            tag = next((t for t in r.get("tags", []) if t.get("in") == "question" and t.get("ann_token")), None)
            if tag is None or tag["ann_token"] not in nusc.t["sample_annotation"]:
                continue
            ann = nusc.t["sample_annotation"][tag["ann_token"]]
            row = {"label": r["answer_label"]}
            for h in HORIZONS:
                for d in ("past", "future"):
                    row[f"{d}{h}"] = yaw_change(nusc, ann, h, d)
            rows.append(row)
        data[split] = rows
    L = ["# Where do DriveLM turn labels come from? (P2, nuScenes box yaw)", "",
         f"Linked P2 questions: train {len(data['train'])}, val {len(data['val'])}. "
         f"Val labels: {dict(Counter(r['label'] for r in data['val']))}.", "",
         "Rule: |heading change| >= threshold -> turn (sign gives left/right), else going ahead. "
         "Threshold tuned on train (balanced accuracy), scored on val. Past and future rows of one horizon use the same "
         "questions (objects tracked h seconds both ways). "
         "For reference, R1 on all val P2: balanced accuracy 33.9, turn recall 1.2; the x8 oversampled model: 46.7 / 34.9.", "",
         "| Window | best threshold (deg) | val balanced acc (%) | val turn recall, right direction (%) | val predicted-turn rate (%) | val n |",
         "| --- | --- | --- | --- | --- | --- |"]
    sign = {}
    for h in HORIZONS:
        for d in ("past", "future"):
            key = f"{d}{h}"
            both = lambda rs: [r for r in rs if r[f"past{h}"] is not None and r[f"future{h}"] is not None]
            grid = [s for s in np.arange(2, 60.5, 1.0)]
            tr = [(score(both(data["train"]), key, t) or {}).get("bal_acc") or 0 for t in grid]
            thr = float(grid[int(np.argmax(tr))])
            v = score(both(data["val"]), key, thr)
            if v:
                L.append(f"| {d}, {h} s | {thr:.0f} | {v['bal_acc']:.1f} | {v['turn_recall']:.1f} | {v['pred_turn_rate']:.1f} | {v['n']} |")
            ch = [(r["label"], r[key]) for r in data["train"] if r[key] is not None and r["label"] != "going_ahead"]
            if ch:
                sign[key] = 100 * np.mean([(c > 0) == (lab == "turn_left") for lab, c in ch if abs(c) >= thr] or [float("nan")])
    L += ["", "Sign check (train turn questions above threshold): share whose heading change has the sign of the label "
          "(left = counter-clockwise). Near 100% = labels use the object's own left/right; near 0% = mirrored.", ""]
    L += [f"- {k}: {v:.1f}%" for k, v in sign.items()]
    L += ["", "Reading: compare the best past window with the best future window. A model that only sees the past "
          "(multi-frame input) cannot beat the past row; the future rows show how much of the label is about what "
          "the object will do next."]
    os.makedirs(os.path.dirname(rel(a.out)), exist_ok=True)
    with open(rel(a.out), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
