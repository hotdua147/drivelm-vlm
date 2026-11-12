"""Gather the latest _report.json of every experiment into results/tables.md (and tables.json)."""
import argparse
import glob
import json
import sys
import os

from _common import rel

MAIN_ORDER = ["b0_prior", "b1_siglip", "b2_zeroshot", "r1_textonly", "r1_6cam", "r1_seed1", "r1_seed2", "r2_grid", "r1_10k", "r1_10k_4ep", "r1_80k"]
ABL_ORDER = ["r1_6cam", "abl_no_camname", "abl_front_only"]


def g(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def fmt(v):
    return "–" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))


def fmt3(v):
    return "–" if v is None else f"{v:.3f}"


MIN_QUESTIONS = 50  # smaller reports are smoke / speed tests (e.g. 10 questions), never results


def latest_reports(runs_dir):
    reps = {}
    for path in sorted(glob.glob(os.path.join(runs_dir, "*", "_report.json"))):
        r = json.load(open(path))
        if (r.get("n") or 0) < MIN_QUESTIONS:
            print(f"skip {path}: only {r.get('n')} questions (smoke/speed test)", file=sys.stderr)
            continue
        reps[r["meta"]["exp"]] = r  # sorted by run id (date first) -> last wins
    return reps


def order(reps, pref):
    return [e for e in pref if e in reps] + sorted(e for e in reps if e not in pref)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--out", default="results/tables.md")
    p.add_argument("--title", default="Results")
    args = p.parse_args()
    allreps = latest_reports(rel(args.runs_dir))
    reps = {k: v for k, v in allreps.items() if not v["meta"].get("eval_set")}      # fixed 700-question val
    p2full = {k: v for k, v in allreps.items() if v["meta"].get("eval_set")}        # every val P2 question
    p1_models = [e for e in reps if reps[e]["meta"].get("model_kind") not in ("prior", "siglip")]  # can answer P1
    L = [f"# {args.title}", "", "All rows: same fixed val ids (scene-level split). F1 at tau = 100 px.", ""]
    L += ["## Table 1: main results", "",
          "| Exp | P1 F1 (%) | Halluc. (%) | P2 status m-F1 (%) | P2o observed m-F1 (%) | P3 category (%) | P4 bal. acc (%) | P5 region F1 (%) | P6 region status m-F1 (%) | Parse-fail (%) | Latency p50 (s) | VRAM (GB) | Warnings |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for e in order(reps, MAIN_ORDER):
        r = reps[e]
        no_tags = r["meta"].get("model_kind") in ("prior", "siglip")  # these baselines cannot answer P1
        L.append("| " + " | ".join([e] + [fmt(x) for x in (
            None if no_tags else g(r, "P1", "tau100", "f1"), None if no_tags else r.get("hallucination_rate"),
            g(r, "P2", "macro_f1"), g(r, "P2o", "macro_f1"), g(r, "P3", "category_accuracy"),
            g(r, "P4", "balanced_accuracy"), g(r, "P5", "f1"), g(r, "P6", "macro_f1"),
            r.get("parse_fail_rate"))] + [fmt3(g(r, "latency_s", "p50")), fmt(r.get("peak_vram_gb"))] +
            ["; ".join(r.get("warnings") or []) or "–"]) + " |")
    L += ["", "Grounding rate (% of predicted P1 tags inside some nuScenes box): " +
          ", ".join(f"{e} {fmt(reps[e].get('grounding_rate'))}" for e in order(reps, MAIN_ORDER)
                    if reps[e]["meta"].get("model_kind") not in ("prior", "siglip"))]
    L += ["", "P1 F1 at every tau:", "", "| Exp | tau=50 | tau=100 | tau=200 |", "| --- | --- | --- | --- |"]
    for e in order({k: reps[k] for k in p1_models}, MAIN_ORDER):
        r = reps[e]
        L.append(f"| {e} | " + " | ".join(fmt(g(r, 'P1', f'tau{t}', 'f1')) for t in (50, 100, 200)) + " |")
    L += ["", "## Table 2: camera ablation", "",
          "| Exp | Cameras | Camera names | P1 F1 CAM_FRONT (%) | P1 F1 other cams (%) | Hallucination (%) | Image tokens |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for e in order({k: reps[k] for k in p1_models}, ABL_ORDER):
        r, m = reps[e], reps[e]["meta"]
        L.append("| " + " | ".join([e, str(len(m.get("cameras") or [])), str(m.get("camera_names")),
                                    fmt(g(r, "P1_by_camera_tau100", "front")), fmt(g(r, "P1_by_camera_tau100", "other")),
                                    fmt(r.get("hallucination_rate")), fmt(r.get("image_tokens_mean"))]) + " |")
    L += ["", "## Table 3: generalisation (P1 F1 at tau = 100 px / hallucination %)", ""]
    subsets = sorted({s for r in reps.values() for s in (r.get("subsets") or {})})
    sub_cols = order({k: reps[k] for k in p1_models}, MAIN_ORDER)
    L += ["| Subset | " + " | ".join(sub_cols) + " |",
          "| --- | " + " | ".join("---" for _ in sub_cols) + " |"]
    for s in subsets:
        cells = []
        for e in sub_cols:
            v = g(reps[e], "subsets", s)
            cells.append("–" if not v else f"{fmt(v['P1_f1_tau100'])} / {fmt(v['hallucination_rate'])} (n={v['n']})")
        L.append(f"| {s} | " + " | ".join(cells) + " |")
    L += ["", "## Table 4: label-noise-aware scores", "",
          "Rows whose DriveLM label contradicts the nuScenes annotation (flag `nusc_disagree`, set by `make data`) are dropped.", "",
          "| Exp | P2 acc all / clean (%) | P2o acc all / clean (%) | P4 bal. acc all / clean (%) | P3 vs DriveLM / vs nuScenes (%) |",
          "| --- | --- | --- | --- | --- |"]
    for e in order(reps, MAIN_ORDER):
        r = reps[e]
        cs = r.get("clean_subset") or {}
        L.append("| " + " | ".join([e,
            f"{fmt(g(r, 'P2', 'accuracy'))} / {fmt(g(cs, 'P2', 'accuracy'))}",
            f"{fmt(g(r, 'P2o', 'accuracy'))} / {fmt(g(cs, 'P2o', 'accuracy'))}",
            f"{fmt(g(r, 'P4', 'balanced_accuracy'))} / {fmt(g(cs, 'P4', 'balanced_accuracy'))}",
            f"{fmt(g(r, 'P3', 'category_accuracy'))} / {fmt(g(r, 'P3', 'category_accuracy_vs_nuscenes'))}"]) + " |")
    L += ["", "## Table 5: P2 (moving status) on every val P2 question", "",
          "Turn recall/precision, exact = right direction; any = any turn. Predicted-turn rate = share of all P2 questions answered with a turn "
          "(a random guesser with that rate has any-turn recall equal to it).", "",
          "| Exp | n | turns | accuracy (%) | balanced acc (%) | macro-F1 (%) | turn recall, exact (%) | turn recall, any (%) | turn precision, exact (%) | turn precision, any (%) | predicted-turn rate (%) |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for e in sorted(p2full):
        q = p2full[e].get("P2") or {}
        L.append("| " + " | ".join([e] + [fmt(q.get(k)) for k in ("n", "n_turn", "accuracy", "balanced_accuracy", "macro_f1",
                                                                    "turn_recall", "turn_any_recall", "turn_precision",
                                                                    "turn_any_precision", "pred_turn_rate")]) + " |")
    ci_cols = [("P1 F1", "P1_f1_tau100", ("P1", "tau100", "f1")), ("Halluc.", "hallucination_rate", ("hallucination_rate",)),
               ("P2 m-F1", "P2_macro_f1", ("P2", "macro_f1")), ("P2o m-F1", "P2o_macro_f1", ("P2o", "macro_f1")),
               ("P3 acc", "P3_category_accuracy", ("P3", "category_accuracy")),
               ("P4 bal. acc", "P4_balanced_accuracy", ("P4", "balanced_accuracy")),
               ("P5 F1", "P5_f1", ("P5", "f1")), ("P6 m-F1", "P6_macro_f1", ("P6", "macro_f1"))]
    L += ["", "## Table 6: 95% bootstrap confidence intervals (fixed val)", "",
          "Percentile bootstrap resampling whole val scenes (B = 1000, seed 0): questions from one scene are correlated, "
          "so this is wider than resampling single questions (`ci95_question` in the JSON). Covers sampling noise of the val set only, "
          "not run-to-run training variance (compare seeds for that). `–` = run evaluated before CIs existed: `make reeval`.", "",
          "| Exp | " + " | ".join(c[0] for c in ci_cols) + " |",
          "| --- | " + " | ".join("---" for _ in ci_cols) + " |"]
    for e in order(reps, MAIN_ORDER):
        r = reps[e]
        no_tags = r["meta"].get("model_kind") in ("prior", "siglip")
        cells = []
        for _, key, path in ci_cols:
            v, c = g(r, *path), (r.get("ci95") or {}).get(key)
            if v is None or (no_tags and key in ("P1_f1_tau100", "hallucination_rate")):
                cells.append("–")
            else:
                cells.append(fmt(v) + (f" [{fmt(c[0])}–{fmt(c[1])}]" if c else ""))
        L.append(f"| {e} | " + " | ".join(cells) + " |")
    L += ["", "## Table 7: score by the camera the question is about (CAM_FRONT vs the other five)", "",
          "Use with the camera ablation: a model that only sees CAM_FRONT should keep its front score and drop to the prior on the others.", "",
          "| Exp | " + " | ".join(f"{k} front / other" for k in ("P2o", "P3", "P4", "P5", "P6")) + " | n front / other (P4) |",
          "| --- | " + " | ".join("---" for _ in range(5)) + " | --- |"]
    keys = {"P2o": "P2o_macro_f1", "P3": "P3_category_accuracy", "P4": "P4_balanced_accuracy", "P5": "P5_f1", "P6": "P6_macro_f1"}
    for e in order(reps, MAIN_ORDER + ABL_ORDER[1:]):
        bc = reps[e].get("by_question_camera") or {}
        if not bc:
            continue
        cells = [f"{fmt(g(bc, k, 'front'))} / {fmt(g(bc, k, 'other'))}" for k in keys.values()]
        n4 = bc.get("P4_balanced_accuracy") or {}
        L.append(f"| {e} | " + " | ".join(cells) + f" | {n4.get('n_front', '–')} / {n4.get('n_other', '–')} |")
    out = rel(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write("\n".join(L) + "\n")
    json.dump(allreps, open(out.replace(".md", ".json"), "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
