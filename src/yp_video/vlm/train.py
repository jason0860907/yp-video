"""LoRA fine-tune Qwen3.5-VL on the 6-second rally / non-rally classification.

Uses the manifest produced by `vlm/build_manifest.py`. For each window, samples
N evenly-spaced frames from the cut video on-the-fly, formats them with the
Qwen3.5-VL chat template, and supervises a single-token answer ("rally" /
"non_rally"). Saves a LoRA adapter under `vlm-checkpoints/<model>/<stamp>/`.

Run from CLI:

    python -m yp_video.vlm.train --model Qwen/Qwen3.5-0.8B \
        --manifest ~/videos/vlm/rally_windows.jsonl \
        --batch-size 4 --epochs 3 --lr 1e-4

Notes:
- Defaults assume one consumer GPU with ~24 GB VRAM. Tune `--batch-size`,
  `--n-frames`, and `--gradient-accumulation` as needed.
- Adapter only is saved (PEFT). Full base weights are not duplicated per run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from yp_video.config import VLM_CHECKPOINTS_DIR, VLM_MANIFEST_FILE
from yp_video.core.jsonl import append_jsonl, write_meta_header


LABELS = ("non_rally", "rally")
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

INSTRUCTION = (
    "You are a volleyball broadcast analyst. Look at the frames sampled from "
    "a 6-second clip of a volleyball match. Decide whether the clip shows a "
    "rally being actively played (the ball is in play between teams). "
    "Respond with exactly one word: \"rally\" or \"non_rally\"."
)


# ── Dataset ──────────────────────────────────────────────────────────────


def _read_manifest(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _sample_frame_indices(num_frames: int, fps: float, start: float, end: float, k: int) -> list[int]:
    """k evenly-spaced frame indices within [start, end]."""
    s_f = max(0, int(round(start * fps)))
    e_f = min(num_frames - 1, int(round(end * fps)))
    if e_f <= s_f:
        return [s_f]
    return [int(round(s_f + (e_f - s_f) * i / max(1, k - 1))) for i in range(k)]


class RallyClipDataset(Dataset):
    """Each item: {frames: list[PIL.Image], label_text: str, label_id: int}.

    Frame extraction is deferred to __getitem__ so the manifest stays cheap to
    load. Uses `decord` for random access into the cut video.
    """

    def __init__(self, records: list[dict], n_frames: int = 8):
        self.records = records
        self.n_frames = n_frames

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        from PIL import Image
        from decord import VideoReader, cpu

        r = self.records[idx]
        vr = VideoReader(r["video"], ctx=cpu(0))
        fps = vr.get_avg_fps() or 30.0
        idxs = _sample_frame_indices(len(vr), fps, r["start"], r["end"], self.n_frames)
        arr = vr.get_batch(idxs).asnumpy()  # (k, H, W, 3) uint8
        frames = [Image.fromarray(a) for a in arr]
        return {"frames": frames, "label_text": r["label"], "label_id": LABEL2ID[r["label"]]}


def _build_chat_messages(frames, label_text: str | None) -> list[dict]:
    """Qwen3-VL-style chat with images. Pass label_text=None at inference."""
    user_content = [{"type": "image", "image": img} for img in frames]
    user_content.append({"type": "text", "text": INSTRUCTION})
    msgs = [{"role": "user", "content": user_content}]
    if label_text is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": label_text}]})
    return msgs


def _make_collator(processor, train: bool = True):
    """Return a collate_fn that materializes a batch ready for processor.__call__.

    Loss is masked everywhere except the assistant-token positions, so the model
    only learns to produce the answer (rally/non_rally), not the prompt.
    """
    def collate(batch):
        all_msgs = [_build_chat_messages(b["frames"], b["label_text"]) for b in batch]

        # Tokenize with images using the processor's chat template
        texts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in all_msgs
        ]
        images = [b["frames"] for b in batch]
        enc = processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        # Build labels: copy input_ids, mask everything before each sample's
        # assistant turn. We approximate "assistant turn start" by the position
        # immediately after applying chat template *without* the assistant turn.
        labels = enc["input_ids"].clone()
        prefix_texts = [
            processor.apply_chat_template(
                _build_chat_messages(b["frames"], None),
                tokenize=False,
                add_generation_prompt=True,
            )
            for b in batch
        ]
        # Find prefix length per sample by re-tokenizing prefix only.
        for i, ptxt in enumerate(prefix_texts):
            prefix_ids = processor.tokenizer(ptxt, add_special_tokens=False)["input_ids"]
            n = min(len(prefix_ids), labels.shape[1])
            labels[i, :n] = -100  # ignore in loss
        # Also mask padding positions
        if processor.tokenizer.pad_token_id is not None:
            labels[enc["input_ids"] == processor.tokenizer.pad_token_id] = -100
        enc["labels"] = labels
        return enc

    return collate


# ── Eval ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(model, processor, val_records, n_frames, max_eval: int = 256, device="cuda") -> dict:
    """Greedy single-token decode → compute binary accuracy / per-source acc."""
    from PIL import Image
    from decord import VideoReader, cpu

    if not val_records:
        return {"n": 0}
    sample = val_records[:max_eval]

    rally_token_ids = processor.tokenizer("rally", add_special_tokens=False)["input_ids"]
    non_token_ids = processor.tokenizer("non_rally", add_special_tokens=False)["input_ids"]

    correct = 0
    by_src = {}
    model.eval()
    for r in sample:
        vr = VideoReader(r["video"], ctx=cpu(0))
        fps = vr.get_avg_fps() or 30.0
        idxs = _sample_frame_indices(len(vr), fps, r["start"], r["end"], n_frames)
        frames = [Image.fromarray(a) for a in vr.get_batch(idxs).asnumpy()]
        msgs = _build_chat_messages(frames, None)
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = processor(text=[text], images=[frames], return_tensors="pt", padding=True).to(device)
        out = model.generate(**enc, max_new_tokens=4, do_sample=False)
        gen_ids = out[0][enc["input_ids"].shape[1]:]
        decoded = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip().lower()
        pred = "rally" if decoded.startswith("rally") else "non_rally"
        ok = (pred == r["label"])
        correct += int(ok)
        bs = by_src.setdefault(r["source"], {"n": 0, "ok": 0})
        bs["n"] += 1
        bs["ok"] += int(ok)
    return {
        "n": len(sample),
        "accuracy": round(correct / len(sample), 4),
        "by_source": {s: {"n": v["n"], "acc": round(v["ok"] / v["n"], 4)}
                      for s, v in by_src.items()},
    }


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", type=str, default="Qwen/Qwen3.5-0.8B",
                   help="HF model id (default: Qwen/Qwen3.5-0.8B, the multimodal instruct variant)")
    p.add_argument("--manifest", type=Path, default=VLM_MANIFEST_FILE)
    p.add_argument("--work-dir", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--n-frames", type=int, default=8)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-samples", type=int, default=256,
                   help="Cap per-epoch eval to this many val windows (else slow).")
    p.add_argument("--balanced-sampler", action="store_true", default=True)
    p.add_argument("--no-balanced-sampler", dest="balanced_sampler", action="store_false")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if not args.manifest.exists():
        print(f"Error: manifest not found at {args.manifest}. "
              f"Run `python -m yp_video.vlm.build_manifest` first.")
        sys.exit(1)

    if not args.work_dir:
        stamp = datetime.now().strftime("%Y-%m%d-%H%M")
        slug = args.model.split("/")[-1].lower()
        args.work_dir = VLM_CHECKPOINTS_DIR / slug / stamp
    args.work_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.work_dir / "train_log.jsonl"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Lazy imports — heavy deps shouldn't load when only --help is invoked.
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoProcessor,
        Trainer,
        TrainingArguments,
    )

    print(f"Loading model: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Data
    records = _read_manifest(args.manifest)
    train_records = [r for r in records if r["subset"] == "training"]
    val_records = [r for r in records if r["subset"] == "validation"]
    print(f"Manifest: train={len(train_records)} val={len(val_records)}")

    train_ds = RallyClipDataset(train_records, n_frames=args.n_frames)

    # Per-source balanced sampler over windows (mirrors TAD weighting idea).
    sampler = None
    sampler_info = None
    if args.balanced_sampler and train_records:
        srcs = [r["source"] for r in train_records]
        counts = Counter(srcs)
        weights = [1.0 / counts[s] for s in srcs]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        sampler_info = {"per_source": dict(counts)}

    write_meta_header(log_path, {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "manifest": str(args.manifest),
        "n_train_windows": len(train_records),
        "n_val_windows": len(val_records),
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout},
        "opt": {"lr": args.lr, "epochs": args.epochs,
                "batch_size": args.batch_size,
                "grad_accum": args.gradient_accumulation,
                "warmup_ratio": args.warmup_ratio},
        "n_frames": args.n_frames,
        "balanced_sampler": bool(args.balanced_sampler),
        "sampler": sampler_info,
    })

    training_args = TrainingArguments(
        output_dir=str(args.work_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        seed=args.seed,
    )

    collator = _make_collator(processor, train=True)

    class _SamplerTrainer(Trainer):
        def _get_train_sampler(self):
            return sampler if sampler is not None else super()._get_train_sampler()

    trainer = _SamplerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
    )

    # Per-epoch eval (manual: HF Trainer's compute_metrics doesn't fit
    # generative single-token classification cleanly).
    class _EpochEvalCallback:
        def __init__(self):
            self.last_epoch_logged = -1

        def __call__(self, args_, state, control, **kwargs):
            ep = int(state.epoch or 0)
            if ep == self.last_epoch_logged:
                return
            self.last_epoch_logged = ep
            metrics = evaluate(model, processor, val_records, args.n_frames,
                               max_eval=args.eval_samples)
            entry = {"epoch": ep, "step": state.global_step, **metrics}
            append_jsonl(log_path, entry)
            print(f"[VLM Eval] epoch={ep} acc={metrics.get('accuracy')} "
                  f"by_source={metrics.get('by_source')}")

    from transformers import TrainerCallback

    class _CB(TrainerCallback):
        def __init__(self):
            self._impl = _EpochEvalCallback()

        def on_epoch_end(self, args_, state, control, **kwargs):
            self._impl(args_, state, control, **kwargs)

    trainer.add_callback(_CB())

    trainer.train()
    # Save final LoRA adapter
    model.save_pretrained(str(args.work_dir / "adapter_final"))
    processor.save_pretrained(str(args.work_dir / "adapter_final"))
    print(f"\nDone. Adapter saved to {args.work_dir / 'adapter_final'}")


if __name__ == "__main__":
    main()
