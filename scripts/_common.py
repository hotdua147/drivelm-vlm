"""Shared CLI helpers for scripts/ (adds src/ to sys.path, loads config with overrides)."""
import argparse
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from dlp.utils.config import load_config, set_dotted  # noqa: E402


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", required=True, help="configs/exp/<name>.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config entries")
    return p


def get_config(args) -> dict:
    cfg = load_config(args.config)
    for kv in args.set:
        k, v = kv.split("=", 1)
        set_dotted(cfg, k, v)
    return cfg


def rel(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def find_run(runs_dir: str, exp: str, need: str = "") -> str:
    """Latest run directory of exactly this experiment (optionally containing ``need``).

    Run ids are ``<MMDD>_<exp>_<data_version>_s<seed>``. The match is exact on ``exp``: a plain
    glob ``*_r1_10k_*`` would also match ``r1_10k_turnx8`` (bug found on 2026-11-10).
    """
    pat = re.compile(r"^\d{4}_" + re.escape(exp) + r"_[A-Za-z0-9]+_s\d+$")
    cands = sorted(d for d in glob.glob(os.path.join(rel(runs_dir), "*"))
                   if os.path.isdir(d) and pat.match(os.path.basename(d))
                   and (not need or os.path.exists(os.path.join(d, need))))
    if not cands:
        raise SystemExit(f"no run found for exp={exp} in {runs_dir} (need {need or 'any'})")
    return cands[-1]


if __name__ == "__main__":  # used by the Makefile: python scripts/_common.py latest <runs_dir> <exp>
    if len(sys.argv) == 4 and sys.argv[1] == "latest":
        print(find_run(sys.argv[2], sys.argv[3]))
    else:
        raise SystemExit("usage: python scripts/_common.py latest <runs_dir> <exp>")
