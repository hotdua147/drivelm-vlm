"""Predict on the fixed val ids and write runs/<run_id>/_preds.jsonl."""
import os
import time

from _common import base_parser, find_run, get_config, rel
from dlp.utils.runs import init_run, read_jsonl, run_id, write_jsonl


def load_val(cfg, max_eval=None):
    P, D = cfg["paths"], cfg["data"]
    proc = rel(P["processed"])
    es = cfg.get("eval_set")
    if es:  # e.g. every val P2 question (the fixed set has only ~9 turns among 100 P2 questions)
        rows = [r for r in read_jsonl(os.path.join(proc, "val.jsonl")) if r["qtype"] in es["qtypes"]]
        rows.sort(key=lambda r: r["id"])
        return rows[:max_eval] if max_eval else rows
    ids_path = os.path.join(proc, f"val_{D['n_val_fixed']}_ids.txt")
    ids = set(open(ids_path).read().split())
    rows = [r for r in read_jsonl(os.path.join(proc, "val.jsonl")) if r["id"] in ids]
    rows.sort(key=lambda r: r["id"])
    return rows[:max_eval] if max_eval else rows


def make_predictor(cfg):
    kind = cfg["model"]["kind"]
    P = cfg["paths"]
    if kind == "prior":
        from dlp.models.prior_baseline import PriorBaseline
        train = read_jsonl(os.path.join(rel(P["processed"]), "train.jsonl"))
        m = PriorBaseline().fit(train)
        return lambda row, cams: {"prediction": m.predict(row)}
    if kind == "mock":
        from dlp.models.mock import NoisyOracle
        m = NoisyOracle(seed=cfg.get("seed", 0))
        return lambda row, cams: {"prediction": m.predict(row)}
    if kind == "siglip":
        from dlp.models.siglip_baseline import SiglipBaseline
        from dlp.eval.profiling import timer, peak_vram_gb, reset_peak
        m = SiglipBaseline(cfg["model"]["model_id"], rel(P["image_root"]), crop=cfg["model"].get("crop", 320))

        def f(row, cams):
            out = {}
            reset_peak()
            with timer(out):
                out["prediction"] = m.predict(row)
            out["peak_vram_gb"] = peak_vram_gb()
            return out
        return f
    if kind == "qwen":
        from dlp.models.qwen_vl import QwenPredictor, load
        M = cfg["model"]
        adapter = None
        src = cfg.get("checkpoint_from") or (cfg["exp"] if cfg.get("train", {}).get("enabled") else None)
        if src:
            adapter = os.path.join(find_run(P["runs_dir"], src, "adapter"), "adapter")
        model, proc = load(M["model_id"], adapter, M["min_pixels"], M["max_pixels"])
        pred = QwenPredictor(model, proc, rel(P["image_root"]), M["mode"],
                             cfg["predict"]["max_new_tokens"], cfg["data"]["cell"])
        shots = few_shot_examples(cfg, int(M.get("few_shot", 0)))
        return lambda row, cams: pred.predict(row, cams, M.get("camera_names", True), shots.get(row["qtype"]))
    raise SystemExit(f"unknown model kind {kind}")


def few_shot_examples(cfg, k: int) -> dict:
    """k fixed training (question, answer) pairs per question type (seed 0), for few-shot prompting."""
    if k <= 0:
        return {}
    import random
    from dlp.utils.runs import read_jsonl
    rows = read_jsonl(os.path.join(rel(cfg["paths"]["processed"]), "train.jsonl"))
    by = {}
    for r in rows:
        by.setdefault(r["qtype"], []).append(r)
    rng = random.Random(0)
    return {qt: [(r["question"], r["answer"]) for r in rng.sample(rs, min(k, len(rs)))] for qt, rs in sorted(by.items())}


def main():
    p = base_parser(__doc__)
    p.add_argument("--max_eval", type=int, default=None)
    p.add_argument("--run_id", default=None)
    args = p.parse_args()
    cfg = get_config(args)
    rid = args.run_id or run_id(cfg["exp"], cfg["data"]["data_version"], cfg.get("seed", 0))
    path = init_run(rel(cfg["paths"]["runs_dir"]), rid, cfg)
    cams = (cfg.get("ablation") or {}).get("cameras") or cfg["data"]["cameras"]
    rows = load_val(cfg, args.max_eval)
    predict = make_predictor(cfg)
    out, t0 = [], time.time()
    for i, row in enumerate(rows):
        res = predict(row, cams)
        out.append({**row, **res})
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(rows)} ({time.time() - t0:.0f}s)")
    write_jsonl(os.path.join(path, "_preds.jsonl"), out)
    print(f"wrote {len(out)} predictions to {path}/_preds.jsonl")
    print(path)


if __name__ == "__main__":
    main()
