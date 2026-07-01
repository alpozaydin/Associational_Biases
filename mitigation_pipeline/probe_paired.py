"""Eyeball sanity check for the paired-view pipeline.

Draws N random items from ``RAFPairedDataset`` and saves each as a side-by-side
PNG (intact ``x`` | redacted ``x'``) plus a small label header. Purely visual --
if the redaction looks right (hair blacked out, face intact for emotion mode),
the Wk3 trainer can consume this dataset unchanged.

    python probe_paired.py \\
        --data-dir $RAF/test \\
        --annotations $ANNO/annotations.json \\
        --hair-dir $ANNO/hair_masks \\
        --out $WORK/paired_samples --n 8 --mode hair+skin
"""

from __future__ import annotations

import argparse
import os
import random

from PIL import Image, ImageDraw, ImageFont

from label_decoding import RAF_EMOTION_LABELS
from paired_dataset import RAFPairedDataset


def save_pair(item, out_path: str) -> None:
    x, xp = item["x"], item["x_prime"]
    W = x.width + xp.width
    H = max(x.height, xp.height) + 24
    canvas = Image.new("RGB", (W, H), (32, 32, 32))
    canvas.paste(x, (0, 24))
    canvas.paste(xp, (x.width, 24))
    draw = ImageDraw.Draw(canvas)
    label = f"{item['img_id']}  gt={RAF_EMOTION_LABELS[item['gt_index']]}"
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((4, 4), label, fill=(220, 220, 220), font=font)
    canvas.save(out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--hair-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--mode", default="hair", choices=["hair", "hair+skin", "hair+bg"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = RAFPairedDataset(args.data_dir, args.annotations, args.hair_dir, mode=args.mode)
    print(f"dataset size: {len(ds)}")
    rng = random.Random(args.seed)
    idxs = rng.sample(range(len(ds)), min(args.n, len(ds)))
    os.makedirs(args.out, exist_ok=True)
    for i, idx in enumerate(idxs):
        item = ds[idx]
        out = os.path.join(args.out, f"pair_{i:02d}_{item['img_id']}.png")
        save_pair(item, out)
        print(f"  {i:02d} idx={idx} gt={RAF_EMOTION_LABELS[item['gt_index']]} -> {out}")


if __name__ == "__main__":
    main()
