"""One-time RAF-DB (basic) reorg into the parent repo's expected layout.

Official RAF-DB 'basic' ships as:
    .../EmoLabel/list_patition_label.txt   # lines: "train_00001.jpg 5"  (label 1-7)
    .../Image/original/<name>.jpg          # full images (background + hair present)
    .../Image/aligned/<name>_aligned.jpg   # tight ~100x100 face crops

Parent repo (``main.py`` raf_run, ``gradcam``) expects:
    DATASET/{train,test}/{1..7}/<image>

Defaults to the ORIGINAL images: Grad-CAM region analysis needs hair/background,
which the aligned crops remove. Label ids 1-7 already match
``utils/__init__.py``::RAF_DB_EMOTIONS order (1=Surprise .. 7=Neutral), so the
dir number is used directly.

    python reorg_raf.py --raf-root /content/RAF_DB --out /content/RAF_DB/DATASET
    # then point RAF_TRAIN at .../DATASET/train (Cell 0d auto-detects it)

Assumes the standard layout above. If your copy differs, tell me the tree.
"""

import argparse
import os
import shutil

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _find(root: str, name: str, *, is_dir: bool) -> str | None:
    """First path under ``root`` whose basename == ``name`` (file or dir)."""
    for dp, dirs, files in os.walk(root):
        pool = dirs if is_dir else files
        if name in pool:
            return os.path.join(dp, name)
    return None


def reorg(raf_root: str, out_root: str, use_aligned: bool) -> None:
    label_file = _find(raf_root, "list_patition_label.txt", is_dir=False)
    if not label_file:
        raise FileNotFoundError(f"list_patition_label.txt not found under {raf_root}")

    img_sub = "aligned" if use_aligned else "original"
    img_dir = _find(raf_root, img_sub, is_dir=True)
    if not img_dir:
        raise FileNotFoundError(f"Image/{img_sub} dir not found under {raf_root}")
    print(f"labels: {label_file}\nimages: {img_dir}")

    placed = missing = 0
    with open(label_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, label = line.split()
            if not label.isdigit() or not (1 <= int(label) <= 7):
                continue
            split = "train" if name.startswith("train") else "test"
            src_name = name
            if use_aligned:  # label file lists originals; aligned add a suffix
                stem, ext = os.path.splitext(name)
                src_name = f"{stem}_aligned{ext}"
            src = os.path.join(img_dir, src_name)
            if not os.path.exists(src):
                missing += 1
                continue
            dst_dir = os.path.join(out_root, split, label)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy(src, os.path.join(dst_dir, src_name))
            placed += 1

    print(f"placed {placed} images ({missing} missing) -> {out_root}")
    if placed == 0:
        raise RuntimeError("0 images placed — check --raf-root / layout")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raf-root", required=True, help="extracted RAF basic root")
    parser.add_argument("--out", required=True, help="output DATASET dir")
    parser.add_argument("--use-aligned", action="store_true", help="use aligned crops (not recommended)")
    args = parser.parse_args()
    reorg(args.raf_root, args.out, args.use_aligned)
