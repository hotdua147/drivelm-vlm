"""Export a random sample of scored answers for hand-checking the automatic scorer (keyword parsing).

Writes a CSV with the question, the DriveLM answer, the model answer and what the scorer parsed from
each. Fill the column ``human_agrees`` with 1 (scorer is right) or 0, then:
    python scripts/export_audit.py --score results/audit_r1.csv
"""
import argparse
import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dlp.data.templates import categories_from_text, category_from_text, normalize_yesno, parse_options, resolve_choice, status_set  # noqa: E402


def parsed(qt, text, question):
    if qt == "P3":
        return str(category_from_text(text))
    if qt == "P5":
        return "|".join(sorted(categories_from_text(text))) or "none"
    if qt == "P4":
        return str(normalize_yesno(resolve_choice(text, parse_options(question))))
    return str(status_set(resolve_choice(text, parse_options(question))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", help="run directory with _preds.jsonl")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--qtypes", default="P2o,P3,P4,P5,P6")
    p.add_argument("--out", default="results/audit.csv")
    p.add_argument("--score", help="score a filled-in CSV instead")
    a = p.parse_args()
    if a.score:
        rows = [r for r in csv.DictReader(open(a.score, encoding="utf-8")) if r["human_agrees"].strip() in ("0", "1")]
        by = {}
        for r in rows:
            by.setdefault(r["qtype"], []).append(int(r["human_agrees"]))
        for q, v in sorted(by.items()):
            print(f"{q}: scorer agrees with human on {sum(v)}/{len(v)} = {100 * sum(v) / len(v):.1f}%")
        allv = [x for v in by.values() for x in v]
        print(f"all: {sum(allv)}/{len(allv)} = {100 * sum(allv) / len(allv):.1f}%")
        return
    qts = a.qtypes.split(",")
    preds = [json.loads(l) for l in open(os.path.join(a.run, "_preds.jsonl"), encoding="utf-8")]
    preds = [r for r in preds if r["qtype"] in qts and r.get("answer_label")]
    rng = random.Random(0)
    per = max(1, a.n // len(qts))
    sample = []
    for q in qts:
        rs = [r for r in preds if r["qtype"] == q]
        sample += rng.sample(rs, min(per, len(rs)))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "qtype", "question", "drivelm_answer", "label_used", "model_answer", "scorer_parsed", "auto_correct", "human_agrees"])
        for r in sample:
            pp = parsed(r["qtype"], r["prediction"], r["question"])
            w.writerow([r["id"], r["qtype"], r["question"], r["answer"], r["answer_label"], r["prediction"], pp,
                        int(pp == str(r["answer_label"])), ""])
    print(f"wrote {len(sample)} rows to {a.out}")


if __name__ == "__main__":
    main()
