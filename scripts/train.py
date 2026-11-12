"""LoRA fine-tuning of Qwen3-VL-2B (R1 / R2). Needs a CUDA GPU, torch, transformers, peft.

Prints LABEL CHECK on the first example, logs sample generations every
``train.sample_every`` steps (collapse shows up there first) and saves the
adapter to runs/<run_id>/adapter.
"""
import inspect
import json
import math
import os
import random

from _common import base_parser, get_config, rel
from dlp.data.dataset import QwenCollator, QwenSFTDataset, label_check, stratified_sample, apply_oversample
from dlp.utils.runs import init_run, read_jsonl, run_id


def main():  # pragma: no cover - GPU only
    p = base_parser(__doc__)
    p.add_argument("--max_train", type=int, default=None)
    p.add_argument("--max_val", type=int, default=None)
    p.add_argument("--run_id", default=None)
    args = p.parse_args()
    cfg = get_config(args)
    import torch
    from transformers import Trainer, TrainerCallback, TrainingArguments
    from dlp.models.qwen_vl import QwenPredictor, add_lora, load

    T, M, P, D = cfg["train"], cfg["model"], cfg["paths"], cfg["data"]
    seed = cfg.get("seed", 0)
    random.seed(seed)
    torch.manual_seed(seed)
    rid = args.run_id or run_id(cfg["exp"], D["data_version"], seed)
    out_dir = init_run(rel(P["runs_dir"]), rid, cfg)

    rows = read_jsonl(os.path.join(rel(P["processed"]), "train.jsonl"))
    rows = apply_oversample(rows, T.get("oversample"))                      # optional, e.g. P2 turns x8
    rows = stratified_sample(rows, args.max_train or T["max_train"], seed)  # balanced question types
    p2 = [r for r in rows if r["qtype"] == "P2"]
    if p2:
        print("P2 labels in the training sample:", {k: sum(r["answer_label"] == k for r in p2) for k in sorted({r["answer_label"] for r in p2})})
    print("train question types:", {k: sum(r["qtype"] == k for r in rows) for k in sorted({r["qtype"] for r in rows})})
    val = read_jsonl(os.path.join(rel(P["processed"]), "val.jsonl"))[: args.max_val or 50]

    model, proc = load(M["model_id"], None, M["min_pixels"], M["max_pixels"])
    if T["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = add_lora(model, cfg["lora"]["r"], cfg["lora"]["alpha"], cfg["lora"]["dropout"],
                     train_vision=cfg["lora"]["train_vision"])
    ds = QwenSFTDataset(rows, proc, rel(P["image_root"]), M["mode"], T["camera_dropout"], D["cell"], seed)

    first = ds[0]
    lc = label_check(first["input_ids"].tolist(), first["labels"].tolist(),
                     lambda ids: proc.tokenizer.decode(ids, skip_special_tokens=False))
    print("LABEL CHECK", json.dumps(lc, ensure_ascii=False))
    json.dump(lc, open(os.path.join(out_dir, "label_check_train.json"), "w"), indent=1, ensure_ascii=False)
    assert rows[0]["answer"].strip()[:20] in lc["supervised_text"], "loss is not on the answer"

    class SampleCallback(TrainerCallback):
        def on_step_end(self, a, state, control, **kw):
            if state.global_step % T["sample_every"] == 0 and state.global_step > 0:
                model.eval()
                pred = QwenPredictor(model, proc, rel(P["image_root"]), M["mode"], 128, D["cell"])
                with open(os.path.join(out_dir, "samples.jsonl"), "a") as f:
                    for r in val[:3]:
                        res = pred.predict(r, D["cameras"], M.get("camera_names", True))
                        f.write(json.dumps({"step": state.global_step, "id": r["id"], "q": r["question"],
                                            "gt": r["answer"], "pred": res["prediction"]}, ensure_ascii=False) + "\n")
                model.train()

    steps_per_epoch = math.ceil(len(ds) / (T["batch_size"] * T["grad_accum"]))
    warmup_steps = max(1, int(T["warmup_ratio"] * steps_per_epoch * T["epochs"]))  # int works in transformers 4 and 5
    wanted = dict(
        output_dir=os.path.join(out_dir, "trainer"), num_train_epochs=T["epochs"],
        per_device_train_batch_size=T["batch_size"], gradient_accumulation_steps=T["grad_accum"],
        learning_rate=T["lr"], warmup_steps=warmup_steps, lr_scheduler_type="cosine",
        bf16=True, logging_steps=T["logging_steps"], save_steps=T["save_steps"], save_total_limit=1,
        report_to=["tensorboard"], remove_unused_columns=False, dataloader_num_workers=2, seed=seed)
    accepted = set(inspect.signature(TrainingArguments.__init__).parameters)
    dropped = sorted(k for k in wanted if k not in accepted)
    if dropped:
        print("WARNING: this transformers version does not accept", dropped, "- ignored")
    targs = TrainingArguments(**{k: v for k, v in wanted.items() if k in accepted})
    print(f"steps/epoch={steps_per_epoch} warmup_steps={warmup_steps}")
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=QwenCollator(proc.tokenizer.pad_token_id), callbacks=[SampleCallback()])
    trainer.train()
    model.save_pretrained(os.path.join(out_dir, "adapter"))
    json.dump(trainer.state.log_history, open(os.path.join(out_dir, "train_log.json"), "w"), indent=1)
    print("peak VRAM (GB):", torch.cuda.max_memory_allocated() / 1024 ** 3)
    print(out_dir)


if __name__ == "__main__":
    main()
