"""Concept-dependent redaction of an image into the paired-view ``x'`` (Roadmap
Sec 3 table).

For emotion (Wk2/3):
  - ``hair``       -> blacken hair region only
  - ``hair+skin``  -> also grayscale (geometry-preserved) the facial-skin area
                     (face bbox minus hair overlap). Removes skin-tone colour
                     without touching expression geometry.

For activity (Wk8-9, later):
  - ``hair+bg``    -> blacken hair + background (everything outside the face
                     bbox); body/pose survive.

The intact view ``x`` is the untouched PIL image, so this module only produces
``x'``.

Hair masks come from ``make_annotations.py`` as ``.npy`` files (float or bool,
saved shape either ``(H, W)`` or ``(H, W, C)``); face bboxes come from
``face_coords`` in the same annotations JSON. Both are in the ORIGINAL image
frame, so redaction happens BEFORE any processor resize.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _load_hair_mask(mask_path: str, size_wh: tuple[int, int]) -> np.ndarray:
    """Return a ``(H, W)`` uint8 hair mask in {0, 1}, resized to ``size_wh``.

    ``size_wh`` is PIL's ``(width, height)``; cv2 resize takes ``(w, h)``.
    """
    mask = np.load(mask_path)
    if mask.ndim == 3:
        mask = mask[..., 0]  # collapse trailing channel if any
    mask = (mask > 0).astype(np.uint8)
    W, H = size_wh
    if mask.shape != (H, W):
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    return mask


def _skin_mask(size_wh: tuple[int, int], face_bbox, hair: np.ndarray) -> np.ndarray:
    """Face bbox minus hair overlap; the region where skin-tone colour lives."""
    W, H = size_wh
    m = np.zeros((H, W), dtype=np.uint8)
    x1, y1, x2, y2 = (int(v) for v in face_bbox)
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(W, x2); y2 = min(H, y2)
    m[y1:y2, x1:x2] = 1
    return np.clip(m - np.logical_and(m, hair).astype(np.uint8), 0, 1)


def _background_mask(size_wh: tuple[int, int], face_bbox, hair: np.ndarray) -> np.ndarray:
    """Everything outside the face bbox AND not hair -- for activity redaction."""
    W, H = size_wh
    inside = np.zeros((H, W), dtype=np.uint8)
    x1, y1, x2, y2 = (int(v) for v in face_bbox)
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(W, x2); y2 = min(H, y2)
    inside[y1:y2, x1:x2] = 1
    return np.clip(1 - np.clip(inside + hair, 0, 1), 0, 1).astype(np.uint8)


def redact(
    image: Image.Image,
    hair_mask_path: str,
    face_bbox=None,
    mode: str = "hair",
) -> Image.Image:
    """Produce ``x'`` from an intact image + hair mask (+ face bbox for skin/bg modes).

    :param image: PIL RGB image at its native resolution.
    :param hair_mask_path: path to a ``.npy`` binary hair mask (resized as needed).
    :param face_bbox: ``[x1, y1, x2, y2]`` in ``image`` coords. Required for
        ``hair+skin`` and ``hair+bg``.
    :param mode: ``hair`` | ``hair+skin`` | ``hair+bg``.
    """
    rgb = np.array(image.convert("RGB"))
    hair = _load_hair_mask(hair_mask_path, image.size)

    rgb = rgb * (1 - hair[..., None])  # blacken hair everywhere

    if mode == "hair":
        return Image.fromarray(rgb)

    if face_bbox is None:
        raise ValueError(f"face_bbox required for mode {mode!r}")

    if mode == "hair+skin":
        skin = _skin_mask(image.size, face_bbox, hair)
        # Luminance-weighted grayscale on skin only; RGB elsewhere untouched.
        gray = (
            rgb[..., 0] * 0.2989 + rgb[..., 1] * 0.5870 + rgb[..., 2] * 0.1140
        ).astype(np.uint8)
        mask_b = skin.astype(bool)
        for c in range(3):
            rgb[..., c] = np.where(mask_b, gray, rgb[..., c])
        return Image.fromarray(rgb)

    if mode == "hair+bg":
        bg = _background_mask(image.size, face_bbox, hair)
        rgb = rgb * (1 - bg[..., None])
        return Image.fromarray(rgb)

    raise ValueError(f"unknown mode {mode!r}; expected 'hair', 'hair+skin', or 'hair+bg'")
