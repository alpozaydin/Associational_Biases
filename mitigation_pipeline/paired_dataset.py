"""RAF paired-view Dataset: yields ``(x, x', gt_index, img_id)`` for Wk3 training.

Expects:
  - ``data_dir`` laid out as ``{1..7}/<image>.jpg`` (dir number is 1-based RAF label).
  - ``annotations_json`` produced by ``make_annotations.py``:
    ``{img_id: {face_1: {face_coords, hair_mask_path}}}``.
  - ``hair_dir`` with the ``.npy`` masks referenced from ``hair_mask_path``.

Images without an annotation (RetinaFace no-detect) are dropped at construction.
Redaction (``x'``) is done per-item at ``__getitem__`` time; hair masks are
small and cv2-resize is cheap, so no caching layer yet. Wire the item into the
LLaVA processor + trainer in Wk3.
"""

from __future__ import annotations

import json
import os

from PIL import Image
from torch.utils.data import Dataset

from redaction import redact

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


class RAFPairedDataset(Dataset):
    """Emotion paired-view dataset. ``gt_index`` is 0-based into
    ``label_decoding.RAF_EMOTION_LABELS``.
    """

    def __init__(
        self,
        data_dir: str,
        annotations_json: str,
        hair_dir: str,
        mode: str = "hair",
    ):
        self.data_dir = data_dir
        self.hair_dir = hair_dir
        self.mode = mode
        with open(annotations_json) as f:
            self.ann = json.load(f)

        samples = []
        for label_num in sorted(os.listdir(data_dir)):
            sub = os.path.join(data_dir, label_num)
            if not (os.path.isdir(sub) and label_num.isdigit()):
                continue
            gt_index = int(label_num) - 1
            if not 0 <= gt_index <= 6:
                continue
            for fname in os.listdir(sub):
                if not fname.lower().endswith(_IMG_EXTS):
                    continue
                img_id = os.path.splitext(fname)[0]
                if img_id not in self.ann:
                    continue
                samples.append((os.path.join(sub, fname), img_id, gt_index))
        if not samples:
            raise FileNotFoundError(
                f"no annotated RAF samples under {data_dir} against {annotations_json}"
            )
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        path, img_id, gt_index = self.samples[idx]
        image = Image.open(path).convert("RGB")
        faces = self.ann[img_id]
        info = next(iter(faces.values()))  # RAF = single face
        hair_mask_path = os.path.join(self.hair_dir, info["hair_mask_path"])
        x_prime = redact(image, hair_mask_path, face_bbox=info["face_coords"], mode=self.mode)
        return {"x": image, "x_prime": x_prime, "gt_index": gt_index, "img_id": img_id}
