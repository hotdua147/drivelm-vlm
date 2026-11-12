"""Export failure cases: P1 samples with the most hallucinated or missed objects."""
import argparse
import json
import os

from _common import rel
from dlp.data.templates import parse_tags
from dlp.eval.matching import ground_tag, match_tags
from dlp.utils.config import load_config
from dlp.utils.runs import read_jsonl
from dlp.viz.draw import draw_sample


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--out", default=None, help="default: <run>/failures")
    args = p.parse_args()
    cfg = load_config(os.path.join(args.run, "config.yaml"))
    gt_dir = os.path.join(rel(cfg["paths"]["processed"]), "gt_boxes")
    out = args.out or os.path.join(args.run, "failures")
    os.makedirs(out, exist_ok=True)
    scored = []
    for r in read_jsonl(os.path.join(args.run, "_preds.jsonl")):
        if r["qtype"] != "P1":
            continue
        boxes = json.load(open(os.path.join(gt_dir, f"{r['sample_token']}.json")))
        pred = [t.to_dict() for t in parse_tags(r["prediction"])]
        gt = [t for t in r["tags"] if t["in"] == "answer"]
        m = match_tags(pred, gt, 100)
        matched = {id(p) for p, _, _ in m["pairs"]}
        hall = sum(ground_tag(t, boxes)["hallucinated"] and id(t) not in matched for t in pred)
        scored.append((hall * 2 + m["fn"] + m["fp"], hall, m, r, boxes, pred, gt))
    scored.sort(key=lambda s: -s[0])
    index = []
    for k, (score, hall, m, r, boxes, pred, gt) in enumerate(scored[: args.n]):
        title = f"{r['id']}  missed={m['fn']}  extra={m['fp']}  hallucinated={hall}  ({r['city']}, {r['condition']})"
        img = draw_sample(rel(cfg["paths"]["image_root"]), r["images"], boxes, gt, pred, title)
        name = f"{k:02d}_{r['id']}.jpg"
        img.save(os.path.join(out, name), quality=85)
        index.append({"file": name, "id": r["id"], "missed": m["fn"], "extra": m["fp"], "hallucinated": hall,
                      "question": r["question"], "answer": r["answer"], "prediction": r["prediction"]})
    json.dump(index, open(os.path.join(out, "index.json"), "w"), indent=1, ensure_ascii=False)
    print(f"wrote {len(index)} failure images to {out} (blue: GT boxes, green circle: GT tag, red cross: prediction)")


if __name__ == "__main__":
    main()
