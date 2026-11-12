"""Check every number printed in the phase-1 report against freshly computed results.

Reads results/report_numbers.json (the report's numbers) and compares with the latest _report.json of
each experiment, results/paired.md, results/label_oracle.md and data/processed/label_check.json.
Prints one line per mismatch and a summary; exit code 1 if anything differs.

    python scripts/verify_report.py           # usually via `make verify`
"""
import json
import os
import re
import sys

from _common import rel  # noqa: F401  (adds src/ to sys.path)
from make_tables import latest_reports

KEYS = {"P1": ("P1", "tau100", "f1"), "P1_tau50": ("P1", "tau50", "f1"), "P1_tau200": ("P1", "tau200", "f1"),
        "halluc": ("hallucination_rate",), "grounding": ("grounding_rate",), "P2o": ("P2o", "macro_f1"),
        "P3": ("P3", "category_accuracy"), "P4": ("P4", "balanced_accuracy"), "P5": ("P5", "f1"), "P6": ("P6", "macro_f1"),
        "parse_fail": ("parse_fail_rate",), "latency_p50": ("latency_s", "p50")}
CI = {"ci_P3": "P3_category_accuracy", "ci_P5": "P5_f1", "ci_P6": "P6_macro_f1"}
CAM = {"P2o": "P2o_macro_f1", "P3": "P3_category_accuracy", "P5": "P5_f1", "P6": "P6_macro_f1"}


def get(d, *path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


class Checker:
    def __init__(self):
        self.n, self.bad = 0, []

    def eq(self, where, expected, got, tol):
        self.n += 1
        if got is None or abs(float(got) - float(expected)) > tol:
            self.bad.append(f"MISMATCH {where}: report says {expected}, results give {got}")


def parse_paired(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        if not line.startswith("| ") or " − " not in line.split("|")[1]:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        out[cells[0]] = cells[1:]
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--expected", default="results/report_numbers.json")
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--results", default="results")
    ap.add_argument("--processed", default="data/processed")
    a = ap.parse_args()
    exp = json.load(open(rel(a.expected), encoding="utf-8"))
    ck = Checker()
    reps = latest_reports(rel(a.runs_dir))
    for e, vals in {**exp.get("reports", {}), **exp.get("p2_full", {})}.items():
        r = reps.get(e)
        if r is None:
            ck.bad.append(f"MISSING run for {e}")
            continue
        for k, v in vals.items():
            if k in CI:
                c = get(r, "ci95", CI[k])
                for i in (0, 1):
                    ck.eq(f"{e} {k}[{i}]", v[i], c[i] if c else None, 0.006)
            elif e.startswith("p2full"):
                ck.eq(f"{e} P2 {k}", v, get(r, "P2", k), 0.006)
            elif k == "latency_p50":
                ck.eq(f"{e} {k}", v, get(r, "latency_s", "p50") or get(r, "latency_p50"), 0.0006)
            else:
                ck.eq(f"{e} {k}", v, get(r, *KEYS[k]), 0.006)
    for e, vals in exp.get("other_cameras", {}).items():
        r = reps.get(e) or {}
        for k, v in vals.items():
            got = get(r, "P1_by_camera_tau100", "other") if k == "P1" else get(r, "by_question_camera", CAM[k], "other")
            ck.eq(f"{e} other-camera {k}", v, got, 0.006)
    paired_path = os.path.join(rel(a.results), "paired.md")
    metrics = ["P1 F1", "Halluc.", "P2o m-F1", "P3 acc", "P4 bal. acc", "P5 F1", "P6 m-F1"]
    if os.path.exists(paired_path):
        rows = parse_paired(paired_path)
        for pair, vals in exp.get("paired", {}).items():
            cells = rows.get(pair)
            if cells is None:
                ck.bad.append(f"MISSING paired row {pair}")
                continue
            for m, (d, lo, hi) in vals.items():
                nums = [float(x) for x in re.findall(r"[+-]?\d+\.\d", cells[metrics.index(m)])]
                for name, want, got in zip(("diff", "lo", "hi"), (d, lo, hi), nums or [None] * 3):
                    ck.eq(f"paired {pair} {m} {name}", want, got, 0.051)
    else:
        ck.bad.append("MISSING results/paired.md (run make paired)")
    lo_path = os.path.join(rel(a.results), "label_oracle.md")
    if os.path.exists(lo_path):
        text = open(lo_path, encoding="utf-8").read()
        for win, v in exp.get("label_oracle", {}).items():
            m = re.search(rf"\| {re.escape(win)} \| \d+ \| ([\d.]+) \|", text)
            ck.eq(f"label oracle {win}", v, float(m.group(1)) if m else None, 0.051)
    else:
        ck.bad.append("MISSING results/label_oracle.md (run make label-oracle)")
    lc_path = os.path.join(rel(a.processed), "label_check.json")
    if os.path.exists(lc_path):
        lc = json.load(open(lc_path))
        lc = lc.get("rates", lc)
        for k, v in exp.get("label_check", {}).items():
            ck.eq(f"label check {k}", v, lc.get(k), 0.00051)
    else:
        print(f"note: {lc_path} not found, label-check numbers not verified (they are printed by make data)")
    for b in ck.bad:
        print(b)
    print(f"\nverify_report: {ck.n - len([b for b in ck.bad if b.startswith('MISMATCH')])}/{ck.n} numbers match"
          + ("" if not ck.bad else f", {len(ck.bad)} problem(s)"))
    sys.exit(1 if ck.bad else 0)


if __name__ == "__main__":
    main()
