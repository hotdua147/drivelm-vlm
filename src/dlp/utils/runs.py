"""Run directories: runs/<date>_<exp>_<data_version>_s<seed>/ with config, git sha, pip freeze."""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys

import yaml


def run_id(exp: str, data_version: str, seed: int, date: str | None = None) -> str:
    date = date or dt.date.today().strftime("%m%d")
    return f"{date}_{exp}_{data_version}_s{seed}"


def _cmd(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:
        return ""


def init_run(runs_dir: str, rid: str, cfg: dict) -> str:
    path = os.path.join(runs_dir, rid)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    with open(os.path.join(path, "git_sha.txt"), "w") as f:
        f.write((_cmd(["git", "rev-parse", "HEAD"]) or "no-git") + "\n")
    with open(os.path.join(path, "pip_freeze.txt"), "w") as f:
        f.write(_cmd([sys.executable, "-m", "pip", "freeze"]) + "\n")
    with open(os.path.join(path, "command.txt"), "w") as f:
        f.write(" ".join(sys.argv) + "\n")
    return path


def read_jsonl(path: str):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
