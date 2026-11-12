"""B1: SigLIP2 zero-shot on a crop around the object tag (P2/P2o status, P3 category).

The crop is a fixed square around the tag point (no ground-truth box is used, so
the baseline sees exactly what the question gives). Candidate prompts follow the
CLIP convention "a photo of ...". Requires ``transformers`` + ``torch``.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

from PIL import Image

from ..data.templates import parse_options, parse_tags

P2_PROMPTS: Dict[str, str] = {
    "Going ahead.": "a photo of a road user moving straight ahead",
    "Turn left.": "a photo of a road user turning left",
    "Turn right.": "a photo of a road user turning right",
    "Stopped.": "a photo of a parked or stopped road user",
}
P2O_PROMPTS: Dict[str, str] = {
    "Moving.": "a photo of a moving road user",
    "Parked.": "a photo of a parked or stationary road user",
}
P3_PROMPTS: Dict[str, str] = {
    "A car.": "a photo of a car", "A truck.": "a photo of a truck", "A bus.": "a photo of a bus",
    "A pedestrian.": "a photo of a pedestrian", "A bicycle.": "a photo of a bicycle",
    "A motorcycle.": "a photo of a motorcycle", "A barrier.": "a photo of a road barrier",
    "A traffic cone.": "a photo of a traffic cone", "A traffic sign.": "a photo of a traffic sign",
}


def crop_around(img: Image.Image, x: float, y: float, size: int = 320) -> Image.Image:
    sx, sy = img.width / 1600.0, img.height / 900.0
    cx, cy, h = x * sx, y * sy, size * sx / 2
    box = (max(0, cx - h), max(0, cy - h), min(img.width, cx + h), min(img.height, cy + h))
    return img.crop(tuple(int(v) for v in box))


class SiglipBaseline:  # pragma: no cover - needs torch + weights
    def __init__(self, model_id: str = "google/siglip2-so400m-patch14-384", image_root: str = ".",
                 device: Optional[str] = None, crop: int = 320):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        from .compat import dtype_kwargs
        self.model = AutoModel.from_pretrained(
            model_id, **dtype_kwargs(torch.bfloat16 if self.device == "cuda" else torch.float32))
        self.model.to(self.device).eval()
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.root, self.crop = image_root, crop

    def _score(self, img: Image.Image, texts: List[str]) -> List[float]:
        inp = self.proc(text=texts, images=img, padding="max_length", max_length=64, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inp)
        return out.logits_per_image[0].float().tolist()

    def predict(self, row: dict) -> str:
        if row["qtype"] not in ("P2", "P2o", "P3"):
            return ""
        tags = parse_tags(row["question"])
        if not tags:
            return ""
        t = tags[0]
        rel = row["images"].get(t.cam)
        if rel is None:
            return ""
        img = crop_around(Image.open(os.path.join(self.root, rel)).convert("RGB"), t.x, t.y, self.crop)
        prompts = {"P2": P2_PROMPTS, "P2o": P2O_PROMPTS, "P3": P3_PROMPTS}[row["qtype"]]
        answers, texts = list(prompts), list(prompts.values())
        scores = self._score(img, texts)
        best = answers[max(range(len(scores)), key=scores.__getitem__)]
        for k, v in parse_options(row["question"]).items():
            if v.strip().lower().rstrip(".") == best.lower().rstrip("."):
                return f"{k}. {v}"
        return best
