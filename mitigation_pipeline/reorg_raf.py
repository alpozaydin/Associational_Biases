"""One-time RAF-DB (basic) reorg into the parent repo's expected layout.

Official RAF-DB 'basic' (as distributed) nests like:
    RAF_DB/basic/basic/EmoLabel/list_patition_label.txt   # "train_00001.jpg 5" (1-7)
    RAF_DB/basic/basic/Image/original.zip                 # full images (bg + hair)
    RAF_DB/basic/basic/Image/aligned.zip                  # tight ~100x100 face crops
plus an __MACOSX/ junk dir. The image sets are *inner zips* that must be
extracted, and the depth varies — so we extract the relevant inner zip, then
index every image basename under the root (skipping __MACOSX) and resolve each
label-file entry by name. Robust to extra nesting / folder-vs-loose layout.

Parent repo (``main.py`` raf_run, ``gradcam``) expects:
    DATASET/{train,test}/{1..7}/<image>

Defaults to ORIGINAL images: Grad-CAM region analysis needs hair/background,
which the aligned crops remove. Label ids 1-7 already match
``utils/__init__.py``::RAF_DB_EMOTIONS (1=Surprise .. 7=Neutral).

    python reorg_raf.py --raf-root /content/RAF_DB --out /content/RAF_DB/DATASET
"""

from __future__ import annotations

import argparse
import os
import shutil
import zipfile

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _find_file(root: str, name: str) -> str | None:
    for dp, _, files in os.walk(root):
        if "__MACOSX" in dp:
            continue
        if name in files:
            return os.path.join(dp, name)
    return None


def _index_images(root: str, skip: str | None = None) -> dict:
    """basename -> full path for every source image under ``root`` (first wins).

    ``skip`` (e.g. the output DATASET dir, which lives under ``root``) is pruned so
    a re-run doesn't index its own output and copy a file onto itself.
    """
    skip_abs = os.path.abspath(skip) if skip else None
    idx = {}
    for dp, dirs, files in os.walk(root):
        if "__MACOSX" in dp or (skip_abs and os.path.abspath(dp).startswith(skip_abs)):
            dirs[:] = []
            continue
        for f in files:
            if f.lower().endswith(_IMG_EXTS):
                idx.setdefault(f, os.path.join(dp, f))
    return idx


def _src_name(label_name: str, use_aligned: bool) -> str:
    if not use_aligned:
        return label_name  # label file already lists originals
    stem, ext = os.path.splitext(label_name)
    return f"{stem}_aligned{ext}"


def reorg(raf_root: str, out_root: str, use_aligned: bool) -> None:
    label_file = _find_file(raf_root, "list_patition_label.txt")
    if not label_file:
        raise FileNotFoundError(f"list_patition_label.txt not found under {raf_root}")
    print("labels:", label_file)

    with open(label_file) as f:
        parts = [ln.split() for ln in f.read().splitlines() if ln.strip()]
    entries = [(p[0], p[1]) for p in parts if len(p) == 2 and p[1].isdigit() and 1 <= int(p[1]) <= 7]
    if not entries:
        raise RuntimeError(f"no valid entries parsed from {label_file}")

    idx = _index_images(raf_root, skip=out_root)
    probe = _src_name(entries[0][0], use_aligned)
    if probe not in idx:  # inner zip not yet extracted -> extract it
        zip_name = "aligned.zip" if use_aligned else "original.zip"
        zpath = _find_file(raf_root, zip_name)
        if not zpath:
            raise FileNotFoundError(f"{zip_name} not found under {raf_root}")
        print("extracting", zpath, "...")
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(os.path.dirname(zpath))
        idx = _index_images(raf_root, skip=out_root)

    placed = missing = 0
    for label_name, label in entries:
        src = idx.get(_src_name(label_name, use_aligned))
        if not src:
            missing += 1
            continue
        split = "train" if label_name.startswith("train") else "test"
        dst_dir = os.path.join(out_root, split, label)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(src, os.path.join(dst_dir, os.path.basename(src)))
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
