"""Demographic-slice accuracy: does the LoRA adapter close, widen, or leave
untouched the acc gap between demographic groups on the probe set?

Advisor call before writing up Wk4b: acc improved != bias reduced. If Wk4b's
+11.8 pp is uniform across gender / age slices, we have a better emotion
classifier and no bias-mitigation story. If concentrated in the
underrepresented / harder slice, that's the paper.

Uses HuggingFace FairFace ports (dima806/fairface_{gender,age}_image_detection)
on the face crop from RetinaFace bbox for demographic labels. Then runs the
same admissible-set decision-token argmax as probe_acc.py for emotion
predictions, and reports per-slice acc.

Usage::

    python demographic_slice.py \\
        --images /path/raf_probe/images \\
        --face-anno /path/raf_probe/anno/annotations.json \\
        --raf-test-root /path/RAF_DB/DATASET/test \\
        --adapter /path/adapters/wk4b_simsiam_hairbg_lambda1

Without ``--adapter``, evaluates the stock backbone. Cached FairFace
predictions are written to ``--demo-cache`` so repeated runs (different
adapters) skip the demographic pass.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from label_decoding import (
    RAF_EMOTION_LABELS,
    _append_prefix,
    build_label_prompt,
    label_decision_set,
)
from lora_describer import DEVICE, _load_backbone, _load_processor

_IMG_EXTS = (".png", ".jpg", ".jpeg")


def _index_labels(raf_test_root: str) -> dict[str, int]:
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


def _crop_face(image: Image.Image, bbox) -> Image.Image:
    x0, y0, x1, y1 = [int(v) for v in bbox]
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(image.width, x1); y1 = min(image.height, y1)
    return image.crop((x0, y0, x1, y1))


def _fairface_predict(models_procs, face_img: Image.Image) -> tuple[str, str]:
    """Returns (gender_label, age_bucket_label)."""
    (mg, pg), (ma, pa) = models_procs
    with torch.no_grad():
        ig = pg(face_img, return_tensors="pt").to(DEVICE)
        gender = mg.config.id2label[int(mg(**ig).logits.argmax(-1))]
        ia = pa(face_img, return_tensors="pt").to(DEVICE)
        age = ma.config.id2label[int(ma(**ia).logits.argmax(-1))]
    return gender, age


def _predict_emotion(processor, model, image, prompt, prefix_ids, label_ids_t) -> int:
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(DEVICE)
    inputs = _append_prefix(inputs, prefix_ids)
    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :]
    return label_ids_t[logits.index_select(-1, label_ids_t).argmax(-1)].item()


def _bucket_age(age: str) -> str:
    """Collapse 9 age buckets to 3 for slice-acc stability (n=40/class -> ~93/bucket)."""
    if age in {"0-2", "3-9", "10-19"}: return "young"
    if age in {"20-29", "30-39", "40-49"}: return "adult"
    return "senior"


def _fill_demo_cache(demo_cache_path: str, images_dir: str, face_anno: str) -> dict[str, dict]:
    if os.path.exists(demo_cache_path):
        with open(demo_cache_path) as f:
            return json.load(f)
    print(f"demographic cache miss, running FairFace on face crops...", flush=True)
    hfk = dict(cache_dir="/mnt/amax5_drive/alp_ozaydin_0/jun/hf_cache")
    mg = AutoModelForImageClassification.from_pretrained(
        "dima806/fairface_gender_image_detection", **hfk).to(DEVICE).eval()
    pg = AutoImageProcessor.from_pretrained(
        "dima806/fairface_gender_image_detection", **hfk)
    ma = AutoModelForImageClassification.from_pretrained(
        "dima806/fairface_age_image_detection", **hfk).to(DEVICE).eval()
    pa = AutoImageProcessor.from_pretrained(
        "dima806/fairface_age_image_detection", **hfk)
    face_anns = json.load(open(face_anno))
    out: dict[str, dict] = {}
    files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(_IMG_EXTS))
    for i, f in enumerate(files):
        stem = os.path.splitext(f)[0]
        anno = face_anns.get(stem)
        if not anno:
            continue
        # first detected face only
        first = next(iter(anno.values()))
        img = Image.open(os.path.join(images_dir, f)).convert("RGB")
        face = _crop_face(img, first["face_coords"])
        gender, age = _fairface_predict(((mg, pg), (ma, pa)), face)
        out[f] = {"gender": gender, "age": age, "age_bucket": _bucket_age(age)}
        if (i + 1) % 40 == 0:
            print(f"  demographic {i + 1}/{len(files)}", flush=True)
    with open(demo_cache_path, "w") as fp:
        json.dump(out, fp, indent=2)
    print(f"demographic cache saved -> {demo_cache_path}", flush=True)
    return out


def evaluate(
    images_dir: str,
    face_anno: str,
    raf_test_root: str,
    demo_cache: str,
    adapter: str | None,
    records_out: str | None = None,
) -> None:
    demo = _fill_demo_cache(demo_cache, images_dir, face_anno)
    labels = _index_labels(raf_test_root)

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

    files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(_IMG_EXTS))
    slice_correct: dict[tuple, int] = defaultdict(int)
    slice_total: dict[tuple, int] = defaultdict(int)
    global_correct = 0
    global_total = 0
    missing = 0
    records: list[dict] = []
    for i, f in enumerate(files):
        gt = labels.get(f)
        dm = demo.get(f)
        if gt is None or dm is None:
            missing += 1
            continue
        img = Image.open(os.path.join(images_dir, f)).convert("RGB")
        pred_tid = _predict_emotion(processor, model, img, prompt, prefix_ids, label_ids_t)
        pred = id2idx[pred_tid]
        ok = int(pred == gt)
        for key in [("gender", dm["gender"]), ("age", dm["age_bucket"]),
                    ("gender+age", f"{dm['gender']}/{dm['age_bucket']}")]:
            slice_total[key] += 1
            slice_correct[key] += ok
        global_total += 1
        global_correct += ok
        records.append({
            "file": f,
            "gt": gt,
            "gt_name": RAF_EMOTION_LABELS[gt],
            "pred": pred,
            "pred_name": RAF_EMOTION_LABELS[pred],
            "gender": dm["gender"],
            "age_bucket": dm["age_bucket"],
            "ok": ok,
        })
        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{len(files)} scored", flush=True)

    print(f"\n===== {tag} =====")
    print(f"global: {global_correct}/{global_total} = {global_correct / global_total:.4f}   (missing: {missing})")
    for axis in ["gender", "age", "gender+age"]:
        print(f"per-slice ({axis}):")
        keys = sorted(k for k in slice_total if k[0] == axis)
        for key in keys:
            n = slice_total[key]
            k = slice_correct[key]
            print(f"  {key[1]:>18s}  n={n:3d}  acc={k / n:.3f}")
    if records_out:
        with open(records_out, "w") as fp:
            json.dump({"tag": tag, "records": records}, fp, indent=2)
        print(f"records saved -> {records_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--face-anno", required=True, help="probe annotations.json (has face_coords per img)")
    ap.add_argument("--raf-test-root", required=True, help="RAF test dir w/ 1..7 subdirs")
    ap.add_argument("--demo-cache", default="/mnt/amax5_drive/alp_ozaydin_0/jun/week1_work/demo_cache.json")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--records", default=None, help="dump per-image (file,gt,pred,gender,age) JSON")
    a = ap.parse_args()
    evaluate(a.images, a.face_anno, a.raf_test_root, a.demo_cache, a.adapter, a.records)
