"""Sample a per-class probe subset from a RAF split into a flat image dir.

Shared by the Colab notebook (Cell 2) and the SSH runner (run_week1.sh) so the
probe set is identical (seeded). RAF filenames are globally unique -> safe to
flatten across the 1..7 class dirs.
"""

import argparse
import os
import random
import shutil

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def make(split_dir: str, out_dir: str, per_class: int, seed: int = 0) -> int:
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    n = 0
    for d in sorted(os.listdir(split_dir)):
        sub = os.path.join(split_dir, d)
        if not (os.path.isdir(sub) and d.isdigit()):
            continue
        files = [f for f in os.listdir(sub) if f.lower().endswith(_IMG_EXTS)]
        rng.shuffle(files)
        for f in files[:per_class]:
            shutil.copy(os.path.join(sub, f), os.path.join(out_dir, f))
            n += 1
    print(f"probe images: {n} -> {out_dir}")
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True, help="RAF split dir (has 1..7 subdirs)")
    ap.add_argument("--out", required=True, help="flat output image dir")
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    make(a.split, a.out, a.per_class, a.seed)
