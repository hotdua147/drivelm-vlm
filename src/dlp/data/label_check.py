"""Label checks collected while building the dataset.

The checker accumulates counters and a bounded list of examples, then
``summary()`` returns the dict written to ``label_check.json``. ``passed`` is
False when a hard threshold is broken; ``build_perception.py`` then exits
non-zero so nothing is trained on broken labels.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Optional

DEFAULT_THRESHOLDS = {
    "min_sample_found_rate": 0.98,   # DriveLM frames found in nuScenes metadata
    "min_tag_valid_rate": 0.99,      # tag inside the image and a known camera
    "min_tag_link_rate": 0.80,       # tag linked to a projected nuScenes box
    "min_status_agreement": 0.70,    # DriveLM moving/stopped vs nuScenes attribute
}


class LabelChecker:
    def __init__(self, thresholds: Optional[Dict[str, float]] = None, max_examples: int = 20):
        self.th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.max_examples = max_examples
        self.c = Counter()
        self.qtype = Counter()
        self.unmatched_questions = Counter()
        self.answers = defaultdict(Counter)
        self.tag_cams = Counter()
        self.examples = defaultdict(list)
        self.p4_templates = defaultdict(Counter)
        self.status_pairs = Counter()
        self.category_confusion = Counter()

    def _ex(self, kind: str, item: dict):
        if len(self.examples[kind]) < self.max_examples:
            self.examples[kind].append(item)

    # -- frame level -------------------------------------------------------
    def frame(self, found: bool, sample_token: str):
        self.c["frames"] += 1
        if found:
            self.c["frames_found"] += 1
        else:
            self._ex("frame_not_in_nuscenes", {"sample_token": sample_token})

    def unverifiable(self, category: Optional[str]):
        self.c["tags_unverifiable"] += 1
        self.c[f"unverifiable:{category}"] += 1

    def tag(self, valid: bool, link: Optional[dict], tag_str: str, sample_token: str, cam: str,
            category: Optional[str] = None):
        self.c["tags"] += 1
        self.c[f"tags_cat:{category}"] += 1
        if link is not None:
            self.c[f"linked_cat:{category}"] += 1
        self.tag_cams[cam] += 1
        if valid:
            self.c["tags_valid"] += 1
        else:
            self._ex("invalid_tag", {"sample_token": sample_token, "tag": tag_str})
        if link is None:
            self._ex("unlinked_tag", {"sample_token": sample_token, "tag": tag_str})
        else:
            self.c["tags_linked"] += 1
            self.c[f"tags_linked_{link['link']}"] += 1

    def status(self, drivelm_coarse: str, nusc_coarse: str, ctx: dict, raw: tuple = ()):
        if raw:
            self.status_pairs[raw] += 1
        if "unknown" in (drivelm_coarse, nusc_coarse):
            self.c["status_unknown"] += 1
            return
        self.c["status_compared"] += 1
        if drivelm_coarse == nusc_coarse:
            self.c["status_agree"] += 1
        else:
            self._ex("status_mismatch", {**ctx, "drivelm": drivelm_coarse, "nuscenes": nusc_coarse})

    def category(self, drivelm_group: Optional[str], nusc_group: str, ctx: dict):
        if drivelm_group in (None, "vehicle"):  # a bare "vehicle" names no nuScenes class
            self.c["category_unspecific"] += 1
            return
        if drivelm_group != nusc_group:
            self.category_confusion[(drivelm_group, ctx.get("nusc_category") or nusc_group)] += 1
        self.c["category_compared"] += 1
        if drivelm_group == nusc_group:
            self.c["category_agree"] += 1
        else:
            self._ex("category_mismatch", {**ctx, "drivelm": drivelm_group, "nuscenes": nusc_group})

    def p4(self, drivelm: str, nusc_by_variant: dict, ctx: dict, parsed: Optional[dict] = None):
        """Agreement of DriveLM yes/no answers with projected nuScenes boxes, per definition variant
        (reported, no threshold: it measures label noise, not a pipeline bug)."""
        if not nusc_by_variant or all(v is None for v in nusc_by_variant.values()):
            self.c["p4_unverifiable"] += 1
            return
        for var, nusc in nusc_by_variant.items():
            self.c[f"p4_compared:{var}"] += 1
            self.c[f"p4_agree:{var}"] += drivelm == nusc
        nusc = nusc_by_variant.get("exact")
        key = f"{(parsed or {}).get('attr') or '-'} {(parsed or {}).get('nusc')}"
        self.p4_templates[key][f"{drivelm}/{nusc}"] += 1
        if drivelm != nusc:
            self._ex("p4_mismatch", {**ctx, "drivelm": drivelm, "nuscenes": nusc})

    def question(self, qtype: str, question: str, answer_label: Optional[str]):
        self.qtype[qtype] += 1
        if qtype == "other":
            self.unmatched_questions[question[:80]] += 1
        if answer_label is not None:
            self.answers[qtype][answer_label] += 1

    # -- summary -------------------------------------------------------------
    @staticmethod
    def _rate(a: int, b: int) -> Optional[float]:
        return round(a / b, 4) if b else None

    def summary(self) -> dict:
        c = self.c
        rates = {
            "sample_found_rate": self._rate(c["frames_found"], c["frames"]),
            "tag_valid_rate": self._rate(c["tags_valid"], c["tags"]),
            "tag_link_rate": self._rate(c["tags_linked"], c["tags"]),
            "tag_link_iou_rate": self._rate(c["tags_linked_iou"], c["tags"]),
            "tag_link_inside_rate": self._rate(c["tags_linked_inside"], c["tags"]),
            "status_agreement": self._rate(c["status_agree"], c["status_compared"]),
            "category_agreement": self._rate(c["category_agree"], c["category_compared"]),
        }
        variants = sorted(k.split(":", 1)[1] for k in c if k.startswith("p4_compared:"))
        p4_rates = {v: self._rate(c[f"p4_agree:{v}"], c[f"p4_compared:{v}"]) for v in variants}
        rates["p4_nuscenes_agreement_by_variant"] = p4_rates
        best = max(p4_rates, key=lambda v: p4_rates[v] or 0) if p4_rates else None
        rates["p4_nuscenes_agreement"] = p4_rates.get(best) if best else None
        cats = sorted(k.split(":", 1)[1] for k in c if k.startswith("tags_cat:"))
        rates["tag_link_rate_by_category"] = {k: self._rate(c[f"linked_cat:{k}"], c[f"tags_cat:{k}"]) for k in cats}
        failures = []
        for key, th in self.th.items():
            name = key.replace("min_", "")
            val = rates.get(name)
            if isinstance(val, (int, float)) and val < th:
                failures.append(f"{name}={val} < {th}")
        p4_tpl = {k: {"n": sum(v.values()),
                      "agree": self._rate(v["yes/yes"] + v["no/no"], sum(v.values())),
                      "drivelm_yes_nusc_no": v["yes/no"], "drivelm_no_nusc_yes": v["no/yes"]}
                  for k, v in sorted(self.p4_templates.items(), key=lambda kv: -sum(kv[1].values()))}
        return {
            "passed": not failures,
            "p4_best_variant": best,
            "p4_by_template": p4_tpl,
            "status_pairs_top": [[list(k), n] for k, n in self.status_pairs.most_common(15)],
            "category_confusion_top": [[list(k), n] for k, n in self.category_confusion.most_common(15)],
            "failures": failures,
            "thresholds": self.th,
            "rates": rates,
            "counts": dict(c),
            "question_types": dict(self.qtype),
            "answer_distribution": {k: dict(v.most_common()) for k, v in self.answers.items()},
            "tag_cameras": dict(self.tag_cams),
            "top_unmatched_questions": dict(self.unmatched_questions.most_common(20)),
            "examples": dict(self.examples),
        }
