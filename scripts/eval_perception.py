"""runs/<id>/_preds.jsonl + gt_boxes -> runs/<id>/_report.json"""
import argparse
import json
import os

from _common import ROOT, rel  # noqa: F401  (adds src/ to sys.path)
from dlp.eval.metrics import evaluate
from dlp.utils.config import load_config
from dlp.utils.runs import read_jsonl


def load_run(run: str, keep_labels: bool = False):
    """(config, predictions, n refreshed fields) of one run."""
    cfg = load_config(os.path.join(run, "config.yaml"))
    preds = read_jsonl(os.path.join(run, "_preds.jsonl"))
    # refresh labels/flags/tags from the current data/processed (labels may have been rebuilt after the
    # run, e.g. a parser fix); the model's prediction is never touched
    refreshed = 0
    if not keep_labels:
        cur = {}
        for split in ("val", "train"):
            path = os.path.join(rel(cfg["paths"]["processed"]), f"{split}.jsonl")
            if os.path.exists(path):
                cur.update({r["id"]: r for r in read_jsonl(path)})
        for r in preds:
            new = cur.get(r["id"])
            if new is not None:
                for k in ("answer_label", "label_flags", "tags", "qtype"):
                    if r.get(k) != new.get(k):
                        r[k] = new[k]
                        refreshed += 1
    return cfg, preds, refreshed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="run directory")
    p.add_argument("--keep_labels", action="store_true", help="score with the labels stored in _preds.jsonl")
    args = p.parse_args()
    cfg, preds, refreshed = load_run(args.run, args.keep_labels)
    if not args.keep_labels:
        print(f"labels refreshed from data/processed: {refreshed} field(s) changed")
    rep = evaluate(preds, os.path.join(rel(cfg["paths"]["processed"]), "gt_boxes"),
                   tuple(cfg["eval"]["taus"]), cfg["eval"]["halluc_dist"])
    vfile = os.path.join(rel(cfg["paths"]["processed"]), "VERSION")
    rep["meta"] = {"exp": cfg["exp"], "model_kind": cfg["model"]["kind"],
                   "data_version": open(vfile).read().strip() if os.path.exists(vfile) else None,
                   "run": os.path.basename(os.path.normpath(args.run)),
                   "labels_refreshed": refreshed,
                   "eval_set": cfg.get("eval_set"),
                   "cameras": (cfg.get("ablation") or {}).get("cameras") or cfg["data"]["cameras"],
                   "mode": cfg["model"].get("mode"), "camera_names": cfg["model"].get("camera_names", True)}
    with open(os.path.join(args.run, "_report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    for w in rep["warnings"]:
        print("WARNING", w)
    print(json.dumps({k: rep.get(k) for k in ("P1", "grounding_rate", "hallucination_rate", "P2", "P3", "P4",
                                             "parse_fail_rate")}, indent=1))


if __name__ == "__main__":
    main()
