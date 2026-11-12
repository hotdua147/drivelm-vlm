"""Prompt building, camera dropout, 3x2 grid and answer-only label masking.

Everything here except ``QwenSFTDataset`` / ``QwenCollator`` is pure Python + PIL
and is unit-tested. The torch classes import torch lazily so the data pipeline
and the tests run on a machine without a GPU stack.
"""
from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Sequence

from PIL import Image

from .templates import CAMERAS

IGNORE_INDEX = -100
GRID_ORDER = [["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"],
              ["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]]

SYSTEM_PROMPT = (
    "You are a driving perception assistant. Objects are referred to as <id,CAMERA,x,y>, "
    "where x,y is the object centre in pixels on the original 1600x900 image of that camera."
)
GRID_NOTE = (
    "The image is a 3x2 grid. Top row: CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT. "
    "Bottom row: CAM_BACK_LEFT, CAM_BACK, CAM_BACK_RIGHT. Always answer with coordinates "
    "of the original camera image, not of the grid."
)


# --------------------------------------------------------------------------- cameras
def referenced_cameras(row: dict) -> set:
    return {t["cam"] for t in row.get("tags", [])}


def choose_cameras(row: dict, cameras: Sequence[str] = CAMERAS, dropout: float = 0.0,
                   rng: Optional[random.Random] = None, keep_only: Optional[Sequence[str]] = None) -> List[str]:
    """Cameras fed to the model. Cameras holding a referenced object are never dropped."""
    if keep_only is not None:
        return [c for c in cameras if c in keep_only]
    if dropout <= 0:
        return list(cameras)
    rng = rng or random
    protected = referenced_cameras(row)
    return [c for c in cameras if c in protected or rng.random() >= dropout]


def load_image(root: str, rel: Optional[str], size=(448, 252)) -> Image.Image:
    if rel is None:
        return Image.new("RGB", size, (0, 0, 0))
    img = Image.open(os.path.join(root, rel)).convert("RGB")
    return img.resize(size, Image.BILINEAR) if size else img


def make_grid(images: Dict[str, Image.Image], cell=(448, 252)) -> Image.Image:
    """3x2 grid in GRID_ORDER; a missing camera is a black cell."""
    w, h = cell
    grid = Image.new("RGB", (3 * w, 2 * h), (0, 0, 0))
    for r, row in enumerate(GRID_ORDER):
        for c, cam in enumerate(row):
            if cam in images:
                grid.paste(images[cam].resize(cell), (c * w, r * h))
    return grid


# --------------------------------------------------------------------------- sampling
def stratified_sample(rows: List[dict], n: int, seed: int = 0, key: str = "qtype") -> List[dict]:
    """Up to n rows with question types as balanced as possible (P1 is only ~2.5% of
    perception QA, so a plain random sample would hardly train it). Types with fewer rows
    than their share give the remainder to the others."""
    rng = random.Random(seed)
    groups: Dict[str, List[dict]] = {}
    for r in rows:
        groups.setdefault(r.get(key), []).append(r)
    for g in groups.values():
        rng.shuffle(g)
    out, left = [], n
    pending = sorted(groups, key=lambda k: len(groups[k]))
    while pending and left > 0:
        share = left // len(pending)
        k = pending.pop(0)
        take = groups[k][: max(share, 0) if pending else left]
        out += take
        left -= len(take)
    rng.shuffle(out)
    return out


def apply_oversample(rows: List[dict], rules: Optional[List[dict]]) -> List[dict]:
    """Repeat rows that match a rule ``factor`` times before sampling, e.g.
    ``{"qtype": "P2", "labels": ["turn_left", "turn_right"], "factor": 8}``. The question-type
    balance of stratified_sample is unchanged; only the mix inside the matched type shifts."""
    if not rules:
        return rows
    out = []
    for r in rows:
        k = 1
        for rule in rules:
            if r.get("qtype") == rule["qtype"] and r.get("answer_label") in rule.get("labels", [r.get("answer_label")]):
                k = max(k, int(rule["factor"]))
        out += [r] * k
    return out


# --------------------------------------------------------------------------- messages
def build_messages(row: dict, image_root: str, mode: str = "separate", cameras: Sequence[str] = CAMERAS,
                   camera_names: bool = True, with_answer: bool = True, cell=(448, 252),
                   examples: Optional[Sequence[tuple]] = None) -> List[dict]:
    """Chat messages in the Qwen-VL format (PIL images inline).

    ``examples``: optional (question, answer) text pairs shown as earlier turns (few-shot, no images),
    so a zero-shot model sees the expected answer style and the tag format <c,CAM,x,y>."""
    content: List[dict] = []
    if mode == "separate":
        for cam in cameras:
            if camera_names:
                content.append({"type": "text", "text": f"{cam}:"})
            content.append({"type": "image", "image": load_image(image_root, row["images"].get(cam), cell)})
    elif mode == "grid":
        imgs = {cam: load_image(image_root, row["images"].get(cam), cell) for cam in cameras}
        content.append({"type": "image", "image": make_grid(imgs, cell)})
        content.append({"type": "text", "text": GRID_NOTE})
    elif mode == "none":  # text-only ablation
        pass
    else:
        raise ValueError(f"unknown image mode {mode}")
    content.append({"type": "text", "text": row["question"]})
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    for q, a in examples or []:
        msgs += [{"role": "user", "content": [{"type": "text", "text": "(Example, images omitted) " + q}]},
                 {"role": "assistant", "content": [{"type": "text", "text": a}]}]
    msgs.append({"role": "user", "content": content})
    if with_answer:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": row["answer"]}]})
    return msgs


# --------------------------------------------------------------------------- label masking
def mask_labels(input_ids: Sequence[int], prompt_len: int, pad_id: Optional[int] = None) -> List[int]:
    """Supervise only tokens after the prompt (the assistant answer + end token)."""
    labels = list(input_ids)
    for i in range(min(prompt_len, len(labels))):
        labels[i] = IGNORE_INDEX
    if pad_id is not None:
        labels = [IGNORE_INDEX if t == pad_id else l for t, l in zip(input_ids, labels)]
    return labels


def label_check(input_ids: Sequence[int], labels: Sequence[int], decode) -> dict:
    """What the loss actually sees; printed by train.py on the first batch (LABEL CHECK)."""
    sup = [t for t, l in zip(input_ids, labels) if l != IGNORE_INDEX]
    return {
        "n_tokens": len(input_ids),
        "n_supervised": len(sup),
        "supervised_text": decode(sup),
    }


# --------------------------------------------------------------------------- torch side
class QwenSFTDataset:  # pragma: no cover - needs torch + transformers
    def __init__(self, rows: List[dict], processor, image_root: str, mode: str = "separate",
                 camera_dropout: float = 0.0, cell=(448, 252), seed: int = 0, max_length: int = 4096):
        self.rows, self.proc, self.root, self.mode = rows, processor, image_root, mode
        self.dropout, self.cell, self.max_length = camera_dropout, tuple(cell), max_length
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        cams = choose_cameras(row, dropout=self.dropout, rng=self.rng)
        full = build_messages(row, self.root, self.mode, cams, with_answer=True, cell=self.cell)
        prompt = full[:-1]
        enc = self.proc.apply_chat_template(full, tokenize=True, return_dict=True, return_tensors="pt")
        p_enc = self.proc.apply_chat_template(prompt, tokenize=True, return_dict=True, return_tensors="pt",
                                              add_generation_prompt=True)
        ids = enc["input_ids"][0].tolist()
        labels = mask_labels(ids, p_enc["input_ids"].shape[1])
        item = {k: v[0] if k in ("input_ids", "attention_mask") else v for k, v in enc.items()}
        import torch
        n = len(ids)
        for k, v in list(item.items()):  # keep token-aligned extras in step with the truncated input_ids
            if torch.is_tensor(v) and v.dim() == 2 and v.shape[0] == 1 and v.shape[1] == n:
                item[k] = v[:, : self.max_length]
        item["labels"] = torch.tensor(labels[: self.max_length])
        item["input_ids"] = item["input_ids"][: self.max_length]
        item["attention_mask"] = item["attention_mask"][: self.max_length]
        return item


class QwenCollator:  # pragma: no cover - needs torch
    """Batch size 1 is the default on 16 GB; for >1, pads text and concatenates vision tensors."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        import torch
        from torch.nn.utils.rnn import pad_sequence
        out = {
            "input_ids": pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=self.pad_id),
            "attention_mask": pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0),
            "labels": pad_sequence([b["labels"] for b in batch], batch_first=True, padding_value=IGNORE_INDEX),
        }
        lengths = [b["input_ids"].shape[0] for b in batch]
        for k in batch[0]:  # vision tensors and any extra model inputs a processor version adds
            if k in out:
                continue
            vals = [b[k] for b in batch]
            if not torch.is_tensor(vals[0]):
                continue
            if len(vals) == 1:
                out[k] = vals[0]
            elif all(v.dim() == 2 and v.shape[0] == 1 and v.shape[1] == n for v, n in zip(vals, lengths)):
                # token-aligned extras, e.g. mm_token_type_ids of shape (1, seq_len) in transformers 5: pad like input_ids
                out[k] = pad_sequence([v[0] for v in vals], batch_first=True, padding_value=0)
            elif all(v.dim() == 1 and v.shape[0] == n for v, n in zip(vals, lengths)):
                out[k] = pad_sequence(vals, batch_first=True, padding_value=0)
            else:  # pixel_values (patches, dim), image_grid_thw (images, 3): concatenate
                out[k] = torch.cat(vals, dim=0)
        return out
