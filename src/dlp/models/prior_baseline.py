"""B0: question-only prior. Answers each template with its most frequent training answer.

It never looks at the images, so any model must beat it clearly on P2/P3/P4
before we can claim the model uses the images. P1 needs coordinates, which a
prior cannot produce sensibly, so it answers P1 with an empty list.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, List

from ..data.templates import parse_options

STATUS_TEXT = {"going_ahead": "Going ahead.", "turn_left": "Turn left.", "turn_right": "Turn right.",
               "stopped": "Stopped.", "reversing": "Reversing.", "other": "Going ahead."}
RIDER_TEXT = {"no_rider": "One bicycle is without a rider.", "rider": "One bicycle is with a rider.",
              "no_rider+rider": "One of the bicycles has a rider, and one does not have a rider."}
COARSE_TEXT_P2O = {"moving": "Moving.", "stopped": "Parked.", "moving+stopped": "Moving.", **RIDER_TEXT}
COARSE_TEXT_P6 = {"moving": "One car is moving.", "stopped": "Many cars are parked.",
                  "moving+stopped": "One car is moving, many cars are parked.", **RIDER_TEXT}
PLURAL = {"car": "cars", "truck": "trucks", "bus": "buses", "pedestrian": "pedestrians", "bicycle": "bicycles",
          "motorcycle": "motorcycles", "barrier": "barriers", "traffic_cone": "traffic cones",
          "traffic_element": "traffic elements"}
GROUP_TEXT = {"car": "A car.", "truck": "A truck.", "bus": "A bus.", "pedestrian": "A pedestrian.",
              "bicycle": "A bicycle.", "motorcycle": "A motorcycle.", "barrier": "A barrier.",
              "traffic_cone": "A traffic cone.", "traffic_element": "A traffic sign."}


def _template_key(q: str) -> str:
    q = re.sub(r"<[^>]+>", "<obj>", q)
    q = q.split("Please select", 1)[0]
    return re.sub(r"\s+", " ", q).strip().lower()[:120]


class PriorBaseline:
    def __init__(self):
        self.by_qtype: Dict[str, Counter] = defaultdict(Counter)
        self.by_template: Dict[str, Counter] = defaultdict(Counter)

    def fit(self, rows: List[dict]) -> "PriorBaseline":
        for r in rows:
            if r["answer_label"] is not None:
                self.by_qtype[r["qtype"]][r["answer_label"]] += 1
                self.by_template[_template_key(r["question"])][r["answer_label"]] += 1
        return self

    def predict(self, row: dict) -> str:
        q = row["qtype"]
        if q == "P1" or q == "other":
            return ""
        counts = self.by_template.get(_template_key(row["question"])) or self.by_qtype.get(q)
        if not counts:
            return ""
        label = counts.most_common(1)[0][0]
        if q == "P2":
            text = STATUS_TEXT.get(label, "Going ahead.")
        elif q == "P2o":
            text = COARSE_TEXT_P2O.get(label, "Moving.")
        elif q == "P6":
            text = COARSE_TEXT_P6.get(label, "Many cars are parked.")
        elif q == "P3":
            text = GROUP_TEXT.get(label, "A car.")
        elif q == "P5":
            groups = [g for g in label.split("|") if g in PLURAL]
            text = ("There are " + " and ".join(PLURAL[g] for g in groups) + ".") if groups else "There are no objects."
        else:
            text = "Yes." if label == "yes" else "No."
        opts = parse_options(row["question"])
        for k, v in opts.items():  # answer with the option letter when the question is multiple choice
            if v.strip().lower().rstrip(".") == text.lower().rstrip("."):
                return f"{k}. {v}"
        return text
