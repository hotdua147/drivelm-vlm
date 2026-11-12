"""MOCK predictor for pipeline smoke tests only. Never report its numbers as results.

It copies the ground-truth answer and corrupts it in controlled ways (moves,
drops and invents tags; flips labels) so every metric path in ``eval`` is
exercised with known, non-trivial values.
"""
from __future__ import annotations

import random

from ..data.templates import Tag, format_tag, parse_options, parse_tags, resolve_choice

FLIP = {"Going ahead.": "Stopped.", "Stopped.": "Going ahead.", "Turn left.": "Going ahead.",
        "Turn right.": "Going ahead.", "Yes.": "No.", "No.": "Yes.", "Moving.": "Parked.", "Parked.": "Moving.",
        " is moving": " is parked", " are parked": " are moving"}


class NoisyOracle:
    def __init__(self, seed: int = 0, jitter_px: float = 40.0, p_drop: float = 0.2,
                 p_invent: float = 0.15, p_flip: float = 0.25):
        self.rng = random.Random(seed)
        self.jitter, self.p_drop, self.p_invent, self.p_flip = jitter_px, p_drop, p_invent, p_flip

    def predict(self, row: dict) -> str:
        a = row["answer"]
        if row["qtype"] == "P1":
            tags = [t for t in parse_tags(a) if self.rng.random() >= self.p_drop]
            moved = [Tag(t.cid, t.cam, t.x + self.rng.gauss(0, self.jitter), t.y + self.rng.gauss(0, self.jitter))
                     for t in tags]
            if self.rng.random() < self.p_invent:
                moved.append(Tag(f"c{len(moved) + 9}", self.rng.choice(["CAM_FRONT", "CAM_BACK"]),
                                 self.rng.uniform(0, 1600), self.rng.uniform(0, 300)))
            return "The important objects are " + ", ".join(format_tag(t) for t in moved) + "."
        a = resolve_choice(a, parse_options(row["question"]))  # 'A.' -> 'Going ahead.'
        if self.rng.random() < self.p_flip:
            for k, v in FLIP.items():
                if k in a:
                    return a.replace(k, v, 1)
            if row["qtype"] == "P3":
                return "A truck."
            if row["qtype"] == "P5":
                return "There are many barriers."
        return a
