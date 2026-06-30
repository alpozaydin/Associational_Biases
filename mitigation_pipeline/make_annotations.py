"""Week-1 annotation builder for the Grad-CAM gate.

Produces the JSON ``gradcam_lora.py`` expects — ``{img_id: {face_1: {face_coords,
hair_mask_path}}}`` — for a RAF probe subset, by reusing the parent explainability
stack (RetinaFace + MediaPipe hair seg) and filling the glue the repo is missing:

  - ``get_face_bboxes.py`` emits RetinaFace ``facial_area`` (+ landmarks).
  - ``get_hair_segmentation.py`` and ``raf_filtering.py`` both read ``face_coords``
    and the hair stage needs an ``image_path`` key per image.

Nothing renames ``facial_area`` -> ``face_coords`` in the repo, so this does it,
then merges face boxes with hair-mask paths. RAF = one face per image.

Pipeline: RetinaFace -> rename/restructure -> hair seg -> merge.

Colab (A100): pip install retina-face mediapipe opencv-python tqdm; needs the
MediaPipe ``hair_segmenter.tflite`` model file (``--hair-model``).

    python make_annotations.py \
        --images   /content/drive/MyDrive/cambridge_bias_mitigation/raf_probe/images \
        --work-dir /content/drive/MyDrive/cambridge_bias_mitigation/raf_probe/anno \
        --hair-model /content/drive/MyDrive/cambridge_bias_mitigation/hair_segmenter.tflite

Outputs under ``--work-dir``:
    faces_raw.json            RetinaFace raw output
    faces_for_hair.json       restructured (face_coords + image_path) -> hair stage input
    hair_masks/               .npy hair masks + hair_masks.json
    annotations.json          <- point gradcam_lora.py --annotations here
                                 (--hair-dir is work-dir/hair_masks)
"""

import argparse
import json
import os
import sys

# Reuse the parent seg stack (explainability_pipeline has no spaces -> importable).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "explainability_pipeline", "retina_face"))
sys.path.insert(0, os.path.join(_ROOT, "explainability_pipeline", "hair_segmentation"))

from get_face_bboxes import get_retina_face_outputs  # noqa: E402
from get_hair_segmentation import get_hair_seg_outputs_raf  # noqa: E402

_IMG_EXTS = (".png", ".jpg", ".jpeg")


def _index_images(images_dir: str) -> dict:
    """Map ``img_id`` (stem) -> filename, matching RetinaFace's output keys."""
    out = {}
    for fname in os.listdir(images_dir):
        if fname.lower().endswith(_IMG_EXTS):
            out[os.path.splitext(fname)[0]] = fname
    if not out:
        raise FileNotFoundError(f"no images under {images_dir}")
    return out


def _restructure_for_hair(raw_path: str, img_index: dict, out_path: str) -> None:
    """RetinaFace raw -> hair-stage input.

    Renames ``facial_area`` -> ``face_coords`` and adds the ``image_path`` key the
    hair stage prepends with ``input_dir``. Skips images where RetinaFace found no
    face (``detect_faces`` returns a non-dict / empty in that case).
    """
    with open(raw_path) as f:
        raw = json.load(f)

    restructured = {}
    for img_id, detection in raw.items():
        if not isinstance(detection, dict) or img_id not in img_index:
            continue
        entry = {"image_path": img_index[img_id]}
        for face_key, face in detection.items():
            if not (isinstance(face, dict) and "facial_area" in face):
                continue
            entry[face_key] = {
                "face_coords": face["facial_area"],
                "gender": "unsure",  # gradcam gate doesn't need gender; placeholder
            }
            break  # RAF = single face
        if len(entry) > 1:  # at least one face added
            restructured[img_id] = entry

    with open(out_path, "w") as f:
        json.dump(restructured, f, indent=2)
    print(f"restructured {len(restructured)} images -> {out_path}")


def _merge(faces_for_hair_path: str, hair_json_path: str, out_path: str) -> None:
    """Merge face_coords (face JSON) with hair_mask_path (hair JSON)."""
    with open(faces_for_hair_path) as f:
        faces = json.load(f)
    with open(hair_json_path) as f:
        hair = json.load(f)

    merged = {}
    for img_id, face_entry in faces.items():
        hair_entry = hair.get(img_id)
        if not hair_entry:
            continue
        out_faces = {}
        for face_key, fdata in face_entry.items():
            if not face_key.startswith("face_"):
                continue
            mask = hair_entry.get(face_key)
            mask_path = mask["hair_mask_path"] if isinstance(mask, dict) else mask
            if not mask_path:
                continue
            out_faces[face_key] = {
                "face_coords": fdata["face_coords"],
                "hair_mask_path": mask_path,
            }
        if out_faces:
            merged[img_id] = out_faces

    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"merged {len(merged)} annotated images -> {out_path}")


def build(images_dir: str, work_dir: str, hair_model: str) -> None:
    os.makedirs(work_dir, exist_ok=True)
    raw_json = os.path.join(work_dir, "faces_raw.json")
    faces_for_hair = os.path.join(work_dir, "faces_for_hair.json")
    hair_dir = os.path.join(work_dir, "hair_masks")
    hair_json = os.path.join(work_dir, "hair_masks", "hair_masks.json")
    annotations = os.path.join(work_dir, "annotations.json")

    print("[1/4] RetinaFace...")
    get_retina_face_outputs(images_dir, work_dir, raw_json, save_images=False)

    print("[2/4] restructure (facial_area -> face_coords)...")
    _restructure_for_hair(raw_json, _index_images(images_dir), faces_for_hair)

    print("[3/4] MediaPipe hair seg...")
    get_hair_seg_outputs_raf(
        input_dir=images_dir,
        output_dir=hair_dir,
        retinaface_json=faces_for_hair,
        output_json=hair_json,
        hair_model=hair_model,
    )

    print("[4/4] merge -> annotations...")
    _merge(faces_for_hair, hair_json, annotations)
    print(f"\nDone. gradcam_lora.py --annotations {annotations} --hair-dir {hair_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="dir of RAF probe images")
    parser.add_argument("--work-dir", required=True, help="output dir for all artifacts")
    parser.add_argument("--hair-model", default="hair_segmenter.tflite", help="MediaPipe .tflite")
    args = parser.parse_args()
    build(args.images, args.work_dir, args.hair_model)
