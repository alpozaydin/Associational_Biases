"""Week-1 LoRA "bite" tune — minimal, just enough to move the adapter off identity.

Purpose is the Sec 9 / Sec 4.3 decision gate, not accuracy: a fresh LoRA adapter
is identity (B zero-init), so Grad-CAM is unchanged until *something* trains. This
runs a short task-only tune (NLL toward the GT RAF emotion at the decision token),
saves the adapter, then ``gradcam_lora.py`` re-runs Grad-CAM to check whether the
vision-tower layer-9 hair activation actually moves under the chosen placement.

This is NOT the real method (no paired views, no consistency loss — those are
Wk2/Wk3, ``train_consistency.py``). Keep it small: a few hundred steps on a RAF
subset is enough to answer "does describer adaptation reach the vision tokens?".

Run (data lives on the Drive root ``cambridge_bias_mitigation`` per colab_runner):
    python bite_tune.py \
        --data-dir /content/drive/MyDrive/cambridge_bias_mitigation/RAF_DB/DATASET/train \
        --placement llm_only --steps 300 --subset 350 \
        --out adapters/bite_llm_only
"""

import argparse
import os
import random

import torch
from PIL import Image

from label_decoding import RAF_EMOTION_LABELS, build_label_prompt, label_decision_set, task_nll
from lora_describer import DEVICE, build_lora_describer, save_adapter

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def collect_raf_samples(data_dir: str):
    """RAF train layout is ``{data_dir}/{1..7}/<image>``; dir number is the label.

    :returns: list of ``(image_path, gt_index)`` where ``gt_index`` is 0-based into
        RAF_EMOTION_LABELS (dir "1" -> Surprise -> index 0).
    """
    samples = []
    for label_num in sorted(os.listdir(data_dir)):
        sub = os.path.join(data_dir, label_num)
        if not (os.path.isdir(sub) and label_num.isdigit()):
            continue
        gt_index = int(label_num) - 1
        if not 0 <= gt_index < len(RAF_EMOTION_LABELS):
            continue
        for fname in os.listdir(sub):
            if fname.lower().endswith(_IMG_EXTS):
                samples.append((os.path.join(sub, fname), gt_index))
    if not samples:
        raise FileNotFoundError(f"no RAF images under {data_dir} (expected 1..7 subdirs)")
    return samples


def _upcast_trainable(model):
    """LoRA params -> fp32 for a stable optimiser step (base stays fp16, frozen).

    Pure-fp16 backward on LoRA is numerically fragile; upcasting only the trainable
    params is the standard cheap fix for a single-GPU spike. For real runs prefer
    bf16 / QLoRA — see TODO.md.
    """
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()


def bite_tune(data_dir, placement, steps, lr, subset, seed, out, labels):
    processor, model = build_lora_describer(placement)
    _upcast_trainable(model)
    model.train()

    prefix_ids, label_ids = label_decision_set(processor, labels)
    prompt = build_label_prompt(processor, labels)

    samples = collect_raf_samples(data_dir)
    rng = random.Random(seed)
    rng.shuffle(samples)
    if subset:
        samples = samples[:subset]
    print(f"{len(samples)} samples, placement={placement}, {steps} steps on {DEVICE}")

    optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)

    step = 0
    running = 0.0
    while step < steps:
        rng.shuffle(samples)
        for img_path, gt_index in samples:
            if step >= steps:
                break
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(DEVICE)

            with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
                loss = task_nll(model, inputs, prefix_ids, label_ids, gt_index)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            running += loss.item()
            step += 1
            if step % 25 == 0:
                print(f"step {step}/{steps}  nll={running / 25:.4f}")
                running = 0.0

    save_adapter(model, out)
    print(f"adapter saved -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="RAF train split root (1..7 subdirs)")
    parser.add_argument(
        "--placement",
        default="llm_only",
        choices=["llm_only", "llm_projector", "llm_projector_vision_late"],
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--subset", type=int, default=350, help="0 = use all samples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="adapters/bite_llm_only")
    args = parser.parse_args()

    bite_tune(
        data_dir=args.data_dir,
        placement=args.placement,
        steps=args.steps,
        lr=args.lr,
        subset=args.subset,
        seed=args.seed,
        out=args.out,
        labels=RAF_EMOTION_LABELS,
    )
