"""raw -> data/processed. Exits 1 when the label check fails (unless --allow_fail)."""
import json
import sys

from _common import base_parser, get_config, rel
from dlp.data.build import build


def main():
    p = base_parser(__doc__)
    p.add_argument("--allow_fail", action="store_true")
    p.add_argument("--report_templates", action="store_true", help="print question-type counts and unmatched questions")
    args = p.parse_args()
    cfg = get_config(args)
    P, D = cfg["paths"], cfg["data"]
    out = build(rel(P["drivelm_json"]), rel(P["nusc_root"]), rel(P["processed"]), P["nusc_version"],
                tuple(D["categories"]), D["cameras"], D["val_frac"], D["seed"], D["n_val_fixed"],
                D.get("thresholds") or {}, D["data_version"])
    lc = out["label_check"]
    print(json.dumps(out["stats"], indent=2))
    print("label check rates:", json.dumps(lc["rates"]))
    if args.report_templates:
        print("question types:", lc["question_types"])
        print("top unmatched:", json.dumps(lc["top_unmatched_questions"], indent=2))
    if not lc["passed"]:
        print("LABEL CHECK FAILED:", lc["failures"], file=sys.stderr)
        if not args.allow_fail:
            sys.exit(1)
    print("LABEL CHECK PASSED" if lc["passed"] else "LABEL CHECK FAILED (allowed)")


if __name__ == "__main__":
    main()
