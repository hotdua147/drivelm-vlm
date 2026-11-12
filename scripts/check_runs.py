"""Integrity check of every run (read-only): are all results from the right model, on the same questions?

Prints, per run: experiment, #questions, hash of the question ids (must be equal across runs scored on
the same set), mean image tokens (588 = 6 cameras, 630 = grid, 98 = front only, 0 = no images), seed,
image mode, training size, adapter source and a hash of the predictions (two runs sharing it gave
identical answers: either the same model or a model that collapsed to one answer).

    python scripts/check_runs.py
"""
import glob
import hashlib
import json
import os
import statistics

from _common import rel  # noqa: F401  (adds src/ to sys.path)
from dlp.utils.config import load_config


def h(s):
    return hashlib.md5(s.encode()).hexdigest()[:8]


def main():
    print(f"{'run':34} {'exp':22} {'n':>5} {'ids':8} {'tok':>4} {'seed':>4} {'mode':8} {'train':>6} {'from':14} {'adapter':7} preds")
    seen = {}
    for d in sorted(glob.glob(os.path.join(rel("runs"), "*/"))):
        p = os.path.join(d, "_preds.jsonl")
        if not os.path.exists(p):
            continue
        rows = sorted((json.loads(l) for l in open(p)), key=lambda r: r["id"])
        if len(rows) < 50:
            continue
        c = load_config(os.path.join(d, "config.yaml"))
        tr = c.get("train", {})
        ph = h("".join(r["prediction"] for r in rows))
        print(f"{os.path.basename(d[:-1]):34} {c['exp']:22} {len(rows):>5} {h(''.join(r['id'] for r in rows))} "
              f"{statistics.mean((r.get('n_image_tokens') or 0) for r in rows):>4.0f} {c.get('seed')!s:>4} "
              f"{str(c['model'].get('mode')):8} {str(tr.get('max_train') if tr.get('enabled') else '-'):>6} "
              f"{str(c.get('checkpoint_from') or '-'):14} {'yes' if os.path.isdir(os.path.join(d, 'adapter')) else '-':7} {ph}")
        if ph in seen:
            n_distinct = len({r["prediction"] for r in rows})
            print(f"  ^ same answers as {seen[ph]} ({n_distinct} distinct answer(s): collapse if 1, else check the adapter)")
        seen.setdefault(ph, c["exp"])


if __name__ == "__main__":
    main()
