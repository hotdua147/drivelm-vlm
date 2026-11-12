"""Paired comparison of experiments on the same val questions (CPU only).

For each pair (A, B) and each headline metric: A - B with a 95% CI from resampling whole val scenes,
both runs on the same resampled questions. A difference is called real when its CI excludes 0.

    python scripts/compare_runs.py                       # default pairs -> results/paired.md
    python scripts/compare_runs.py --pairs r1_6cam:r2_grid r1_6cam:r1_seed1
"""
import argparse
import os

from _common import find_run, rel  # noqa: F401  (adds src/ to sys.path)
from eval_perception import load_run
from dlp.eval.metrics import evaluate, paired_bootstrap

DEFAULT = ["r1_6cam:b0_prior", "r1_6cam:r1_textonly", "r1_textonly:b0_prior", "r1_6cam:r2_grid",
           "r1_6cam:r1_seed1", "r1_6cam:r1_10k_4ep", "r1_10k_4ep:r1_10k", "r1_6cam:abl_front_only"]
METRICS = [("P1 F1", "P1_f1_tau100"), ("Halluc.", "hallucination_rate"), ("P2o m-F1", "P2o_macro_f1"),
           ("P3 acc", "P3_category_accuracy"), ("P4 bal. acc", "P4_balanced_accuracy"), ("P5 F1", "P5_f1"),
           ("P6 m-F1", "P6_macro_f1")]


def units_of(runs_dir, exp, cache):
    if exp not in cache:
        run = find_run(runs_dir, exp, "_preds.jsonl")
        cfg, preds, _ = load_run(run)
        rep = evaluate(preds, os.path.join(rel(cfg["paths"]["processed"]), "gt_boxes"),
                       tuple(cfg["eval"]["taus"]), cfg["eval"]["halluc_dist"], return_units=True)
        cache[exp] = rep["_units"]
    return cache[exp]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--pairs", nargs="*", default=DEFAULT)
    p.add_argument("--out", default="results/paired.md")
    a = p.parse_args()
    cache, L = {}, ["# Paired comparisons (same val questions, scene bootstrap, B = 1000)", "",
                    "Cell = A − B [95% CI]. **Bold** = CI excludes 0 (a real difference on this val set). "
                    "Run-to-run training variance is not included: compare with the r1_6cam − r1_seed1 row.", "",
                    "| A − B | " + " | ".join(m[0] for m in METRICS) + " |",
                    "| --- | " + " | ".join("---" for _ in METRICS) + " |"]
    for pair in a.pairs:
        ea, eb = pair.split(":")
        try:
            ua, ub = units_of(a.runs_dir, ea, cache), units_of(a.runs_dir, eb, cache)
        except SystemExit as e:
            print(f"skip {pair}: {e}")
            continue
        cells = []
        for _, key in METRICS:
            r = paired_bootstrap(ua[key], ub[key]) if key in ua and key in ub else None
            if r is None:
                cells.append("–")
                continue
            lo, hi = r["ci95"]
            txt = f"{r['diff']:+.1f} [{lo:+.1f}, {hi:+.1f}]"
            cells.append(f"**{txt}**" if lo > 0 or hi < 0 else txt)
        L.append(f"| {ea} − {eb} | " + " | ".join(cells) + " |")
        print(L[-1])
    os.makedirs(os.path.dirname(rel(a.out)), exist_ok=True)
    with open(rel(a.out), "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
