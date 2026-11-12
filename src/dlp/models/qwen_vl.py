"""Qwen3-VL-2B loading, LoRA and generation (B2, R1, R2).

Requires ``torch``, ``transformers`` (a version that ships Qwen3-VL) and ``peft``.
Day-1 check: ``python -c "from dlp.models.qwen_vl import print_modules; print_modules()"``
prints the linear layer names so the LoRA regex below can be confirmed.
"""
from __future__ import annotations

from typing import List, Optional

from ..data.dataset import build_messages
from ..eval.profiling import peak_vram_gb, reset_peak, timer

DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
# LoRA only on the language model's attention + MLP projections (not the vision tower)
LM_LORA_REGEX = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


def load(model_id: str = DEFAULT_MODEL, adapter: Optional[str] = None, min_pixels: int = 64 * 32 * 32,
         max_pixels: int = 448 * 252, attn: str = "sdpa"):  # pragma: no cover
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from .compat import dtype_kwargs, set_image_pixels
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, attn_implementation=attn, device_map="auto", **dtype_kwargs(torch.bfloat16))
    proc = AutoProcessor.from_pretrained(model_id)
    print("image pixel budget:", set_image_pixels(proc, min_pixels, max_pixels))
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, proc


def add_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05,
             target_regex: str = LM_LORA_REGEX, train_vision: bool = False):  # pragma: no cover
    from peft import LoraConfig, get_peft_model
    regex = target_regex
    if train_vision:  # optional ablation: also adapt the vision encoder blocks
        regex = rf"({target_regex})|(.*visual.*\.(qkv|proj|linear_fc1|linear_fc2)$)"
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=regex,
                     bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


def print_modules(model_id: str = DEFAULT_MODEL):  # pragma: no cover
    import torch
    model, _ = load(model_id)
    names = sorted({n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)})
    print(names)


class QwenPredictor:  # pragma: no cover
    def __init__(self, model, processor, image_root: str, mode: str = "separate",
                 max_new_tokens: int = 256, cell=(448, 252)):
        self.model, self.proc, self.root, self.mode = model, processor, image_root, mode
        self.max_new_tokens, self.cell = max_new_tokens, tuple(cell)
        tok = processor.tokenizer
        self.image_token_id = tok.convert_tokens_to_ids("<|image_pad|>")

    def predict(self, row: dict, cameras: List[str], camera_names: bool = True, examples=None) -> dict:
        import torch
        msgs = build_messages(row, self.root, self.mode, cameras, camera_names, with_answer=False, cell=self.cell,
                              examples=examples)
        inputs = self.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors="pt").to(self.model.device)
        n_img = int((inputs["input_ids"] == self.image_token_id).sum())
        out: dict = {}
        reset_peak()
        with torch.no_grad(), timer(out):
            gen = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        new = gen[0, inputs["input_ids"].shape[1]:]
        out.update(prediction=self.proc.batch_decode([new], skip_special_tokens=True)[0].strip(),
                   n_image_tokens=n_img, n_new_tokens=int(new.shape[0]), peak_vram_gb=peak_vram_gb())
        return out
