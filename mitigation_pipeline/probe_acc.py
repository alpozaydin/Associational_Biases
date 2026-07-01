"""Probe-set intact-image accuracy for stock vs LoRA adapters.

Sanity check for the noise-vs-signal question on the Grad-CAM CAM deltas: if
CAM shifts but accuracy tanks, the model is being broken; if CAM stays flat
but accuracy holds, adapters aren't destructive.

Reads the probe image dir (flat, RAF filenames), reconstructs each image's
label by looking it up under ``--raf-test-root/{1..7}/<filename>``, runs the
same admissible-set decision-token argmax as ``gradcam_lora.py`` line 178-181.
Prints top-1 accuracy + per-class breakdown.

Usage::

    python probe_acc.py \\
        --images /path/raf_probe/images \\
        --raf-test-root /path/RAF_DB/DATASET/test \\
        [--adapter /path/adapters/wk4_simsiam_lambda1]

Without ``--adapter``, evaluates the stock frozen backbone.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict

import torch
from peft import PeftModel
from PIL import Image

from label_decoding import (
    RAF_EMOTION_LABELS,
    _append_prefix,
    build_label_prompt,
    label_decision_set,
)
from lora_describer import DEVICE, _load_backbone, _load_processor

_IMG_EXTS = (".png", ".jpg", ".jpeg")


def _index_labels(raf_test_root: str) -> dict[str, int]:
    """RAF class dirs (1..7) -> {filename: 0-based label index}."""
    out: dict[str, int] = {}
    for d in sorted(os.listdir(raf_test_root)):
        sub = os.path.join(raf_test_root, d)
        if not (os.path.isdir(sub) and d.isdigit()):
            continue
        cls = int(d) - 1
        for f in os.listdir(sub):
            if f.lower().endswith(_IMG_EXTS):
                out[f] = cls
    return out


def _predict(processor, model, image, prompt, prefix_ids, label_ids_t) -> int:
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(DEVICE)
    inputs = _append_prefix(inputs, prefix_ids)
    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :]
    return label_ids_t[logits.index_select(-1, label_ids_t).argmax(-1)].item()


def evaluate(images_dir: str, raf_test_root: str, adapter: str | None) -> None:
    processor = _load_processor()
    model = _load_backbone(to_device=True).eval()
    tag = "stock"
    if adapter:
        model = PeftModel.from_pretrained(model, adapter).to(DEVICE).eval()
        tag = f"adapter={os.path.basename(adapter)}"

    prefix_ids, label_ids = label_decision_set(processor)
    prompt = build_label_prompt(processor)
    label_ids_t = torch.as_tensor(label_ids, device=DEVICE)
    id2idx = {tid: i for i, tid in enumerate(label_ids)}

    labels = _index_labels(raf_test_root)
    files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(_IMG_EXTS))
    per_class_correct: Counter[int] = Counter()
    per_class_total: Counter[int] = Counter()
    confusion: dict[int, Counter[int]] = defaultdict(Counter)
    missing = 0

    for i, f in enumerate(files):
        gt = labels.get(f)
        if gt is None:
            missing += 1
            continue
        img = Image.open(os.path.join(images_dir, f)).convert("RGB")
        pred_tid = _predict(processor, model, img, prompt, prefix_ids, label_ids_t)
        pred = id2idx[pred_tid]
        per_class_total[gt] += 1
        confusion[gt][pred] += 1
        if pred == gt:
            per_class_correct[gt] += 1
        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{len(files)} scored", flush=True)

    tot = sum(per_class_total.values())
    correct = sum(per_class_correct.values())
    print(f"\n===== {tag} =====")
    print(f"top-1 acc: {correct}/{tot} = {correct / tot:.4f}   (missing labels: {missing})")
    print("per-class:")
    for c in sorted(per_class_total):
        n = per_class_total[c]
        k = per_class_correct[c]
        print(f"  {RAF_EMOTION_LABELS[c]:>10s}  n={n:3d}  acc={k / n:.3f}")
    # Confusion matrix: rows=GT, cols=pred. Diagnoses decoder fallback bias
    # (e.g. a column that soaks up wrong-GT images = default-class pathology).
    classes = sorted(per_class_total)
    header = "  gt\\pred  " + " ".join(f"{RAF_EMOTION_LABELS[c][:4]:>5s}" for c in classes) + "   total"
    print("confusion (rows=GT, cols=pred):")
    print(header)
    for gt in classes:
        row = " ".join(f"{confusion[gt][p]:>5d}" for p in classes)
        print(f"  {RAF_EMOTION_LABELS[gt][:8]:>8s}  {row}   {per_class_total[gt]:>5d}")
    col_totals = {p: sum(confusion[gt][p] for gt in classes) for p in classes}
    col_row = " ".join(f"{col_totals[p]:>5d}" for p in classes)
    print(f"  {'total':>8s}  {col_row}   {tot:>5d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="probe images dir (flat, RAF filenames)")
    ap.add_argument("--raf-test-root", required=True, help="RAF test dir with 1..7 subdirs")
    ap.add_argument("--adapter", default=None, help="optional LoRA adapter dir")
    a = ap.parse_args()
    evaluate(a.images, a.raf_test_root, a.adapter)
