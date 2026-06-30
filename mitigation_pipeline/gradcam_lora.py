"""Week-1 decision gate: does describer LoRA adaptation move vision-tower Grad-CAM?

Roadmap Sec 4.3 / Sec 9 — the single highest-risk assumption. If we adapt only
the LLM tower and the CLIP vision tokens still encode hair/skin, the layer-9
Grad-CAM hair activation will NOT drop and the RQ-B mechanistic claim collapses;
the answer then is to escalate placement (``llm_projector`` -> ``+vision_late``).

This re-runs the parent's Grad-CAM probe (``vision_tower...layers[9].layer_norm2``,
token-conditioned, region-normalised) on a **LoRA** describer and, with
``--compare``, on the stock backbone too, printing the hair/face/background delta.
Machinery (``reshape_transform``, ``TokenLogitWrapper``, region masks) is lifted
verbatim from ``explainability_pipeline/gradcam_pipeline/run_gradcam_pipeline.ipynb``
so numbers stay comparable to the paper.

Targeting differs from the notebook in one principled way: instead of free-form
generating and matching a keyword, we read the **decision token** (last input
position) and attribute the *chosen admissible label's* logit — the same
token-selection rule, made deterministic (see label_decoding.py).

Requires the RetinaFace + hair-mask annotation JSON the explainability pipeline
emits (``face_coords`` per face, ``hair_mask_path``). CUDA recommended (fp16
Grad-CAM backward).
"""

import argparse
import json
import os

import cv2
import numpy as np
import torch
from peft import PeftModel
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from label_decoding import (
    RAF_EMOTION_LABELS,
    _append_prefix,
    build_label_prompt,
    label_decision_set,
)
from lora_describer import DEVICE, load_adapter_describer
from lora_describer import _load_backbone, _load_processor  # stock baseline for --compare

_IMG_EXTS = (".png", ".jpg", ".jpeg")


# --- machinery lifted from run_gradcam_pipeline.ipynb (keep identical) ---------
def reshape_transform(tensor):
    B, N, C = tensor.shape
    H_W = int((N - 1) ** 0.5)
    assert H_W * H_W == (N - 1), f"Cannot reshape {N - 1} tokens into square feature map"
    tensor = tensor[:, 1:, :]
    return tensor.reshape(B, H_W, H_W, C).permute(0, 3, 1, 2)


class TokenLogitWrapper(torch.nn.Module):
    def __init__(self, base_model, inputs_template, token_index, num_image_views=5):
        super().__init__()
        self.model = base_model
        self.base_inputs = inputs_template.copy()
        self.base_inputs.pop("pixel_values", None)
        self.base_inputs.pop("image_sizes", None)
        self.token_index = token_index
        self.num_image_views = num_image_views

    def forward(self, pixel_values_4d):
        current_inputs = self.base_inputs.copy()
        B, C, H, W = pixel_values_4d.shape
        pixel_values_5d = torch.zeros(
            B, self.num_image_views, C, H, W,
            dtype=pixel_values_4d.dtype, device=pixel_values_4d.device,
        )
        pixel_values_5d[:, 0, :, :, :] = pixel_values_4d
        current_inputs["pixel_values"] = pixel_values_5d
        current_inputs["image_sizes"] = torch.tensor([(H, W)] * B, device=pixel_values_4d.device)
        out = self.model(**current_inputs)
        logits = out.logits
        target_logits = logits[:, self.token_index, :]
        return target_logits.unsqueeze(0) if target_logits.ndim == 1 else target_logits


def upscale_image_if_needed(img, target_size=336):
    if min(img.size) < target_size:
        return img.resize((target_size, target_size), Image.BICUBIC)
    return img


def rect_to_mask(image_shape, rect):
    x1, y1, x2, y2 = map(int, rect)
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask


def scale_bbox(bbox, orig_size, new_size):
    x_scale = new_size[0] / orig_size[0]
    y_scale = new_size[1] / orig_size[1]
    x, y, w, h = bbox
    return [int(x * x_scale), int(y * y_scale), int(w * x_scale), int(h * y_scale)]


def average_activation(cam_map, region_mask):
    area = np.sum(region_mask)
    return 0.0 if area == 0 else np.sum(cam_map * region_mask) / area
# --- end lifted machinery ------------------------------------------------------


def _llava(model):
    """Unwrap to the LlavaNext module whether ``model`` is stock or a PeftModel.

    ``PeftModel.get_base_model()`` returns the wrapped LlavaNext; a stock model is
    already it. (Don't use ``PreTrainedModel.base_model`` — that returns the inner
    transformer, which has no ``.vision_tower``.)
    """
    return model.get_base_model() if isinstance(model, PeftModel) else model


def _vision_tower(llava):
    """Locate the CLIP vision tower across transformers versions.

    transformers >=4.48 nests it under ``.model`` (LlavaNextModel) instead of
    directly on the ForConditionalGeneration class.
    """
    if hasattr(llava, "vision_tower"):
        return llava.vision_tower
    return llava.model.vision_tower


def gradcam_for_label(processor, model, image, prompt, prefix_ids, label_ids):
    """Grad-CAM heatmap attributing the chosen admissible label to vision features.

    :returns: ``(grayscale_cam, label_name, orig_size)``. ``grayscale_cam`` is HxW
        in the upscaled-image frame (region masks must be resized to it).
    """
    orig_size = image.size
    image = upscale_image_if_needed(image)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(DEVICE)
    # Feed the shared prefix (e.g. the leading-space piece) so position -1 predicts
    # the discriminative content token, then pick the predicted admissible label.
    inputs = _append_prefix(inputs, prefix_ids)

    with torch.no_grad():
        last = model(**inputs).logits[:, -1, :]
        idx = torch.as_tensor(label_ids, device=last.device)
        chosen = idx[last.index_select(-1, idx).argmax(-1)].item()
    label_name = RAF_EMOTION_LABELS[label_ids.index(chosen)]

    num_views = inputs["pixel_values"].shape[1]
    pixel_4d = inputs["pixel_values"][:, 0].clone().detach().requires_grad_(True)
    token_index = inputs["input_ids"].shape[1] - 1
    target_layer = _vision_tower(_llava(model)).vision_model.encoder.layers[9].layer_norm2

    wrapper = TokenLogitWrapper(
        model,
        {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]},
        token_index=token_index,
        num_image_views=num_views,
    )
    cam = GradCAM(model=wrapper, target_layers=[target_layer], reshape_transform=reshape_transform)
    grayscale = cam(input_tensor=pixel_4d, targets=[ClassifierOutputTarget(chosen)])[0]
    return grayscale, label_name, orig_size


def regional_activations(grayscale, orig_size, faces, hair_dir):
    """Mean normalised Grad-CAM over hair / face / background for each annotated face.

    Mirrors the notebook: hair from the mediapipe mask, face from the RetinaFace
    bbox (minus hair overlap), background = remainder; normalised to sum to 1.
    """
    out = []
    h, w = grayscale.shape
    vis_size = (w, h)
    for face_id, info in faces.items():
        hair_path = os.path.join(hair_dir, info["hair_mask_path"])
        if not os.path.exists(hair_path):
            continue
        hair = (np.load(hair_path) > 0).astype(np.uint8)
        hair = cv2.resize(hair, (w, h), interpolation=cv2.INTER_NEAREST)
        face = rect_to_mask(grayscale.shape, scale_bbox(info["face_coords"], orig_size, vis_size))
        face = np.clip(face - np.logical_and(face, hair).astype(np.uint8), 0, 1)
        bg = 1 - np.clip(hair + face, 0, 1)
        acts = {
            "hair": average_activation(grayscale, hair),
            "face": average_activation(grayscale, face),
            "background": average_activation(grayscale, bg),
        }
        total = sum(acts.values())
        if total == 0:
            continue
        out.append({"face_id": face_id, **{k: v / total for k, v in acts.items()}})
    return out


def _load_stock():
    processor = _load_processor()
    return processor, _load_backbone(to_device=True).eval()


def run(images_dir, annotations, hair_dir, adapter, compare, limit):
    with open(annotations) as f:
        ann = json.load(f)

    proc_a, model_a = (load_adapter_describer(adapter) if adapter else _load_stock())
    prefix_ids, label_ids = label_decision_set(proc_a)  # same tokenizer for stock + adapted
    prompt = build_label_prompt(proc_a)
    models = [("adapted" if adapter else "stock", proc_a, model_a)]
    if compare and adapter:
        models.append(("stock", *_load_stock()))

    names = [n for n in sorted(os.listdir(images_dir)) if n.lower().endswith(_IMG_EXTS)]
    if limit:
        names = names[:limit]

    agg = {tag: {"hair": [], "face": [], "background": []} for tag, *_ in models}
    for name in names:
        key = os.path.splitext(name)[0]
        if key not in ann:
            continue
        image = Image.open(os.path.join(images_dir, name)).convert("RGB")
        for tag, proc, model in models:
            gray, _, orig = gradcam_for_label(proc, model, image, prompt, prefix_ids, label_ids)
            for row in regional_activations(gray, orig, ann[key], hair_dir):
                for region in agg[tag]:
                    agg[tag][region].append(row[region])

    print("\n=== mean region activation (Grad-CAM, layer 9, region-normalised) ===")
    for tag in agg:
        n = len(agg[tag]["hair"])
        means = {r: (np.mean(v) if v else float("nan")) for r, v in agg[tag].items()}
        print(f"{tag:>8}  n={n:>4}  hair={means['hair']:.3f}  "
              f"face={means['face']:.3f}  background={means['background']:.3f}")
    if len(agg) == 2:
        a, b = list(agg)  # adapted, stock
        dh = np.mean(agg[a]["hair"]) - np.mean(agg[b]["hair"])
        print(f"\nGATE: hair activation delta ({a} - {b}) = {dh:+.3f} "
              f"({'DROPS -> gate PASS' if dh < 0 else 'does NOT drop -> escalate placement'})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="dir of images to probe")
    parser.add_argument("--annotations", required=True, help="RetinaFace+hair annotation JSON")
    parser.add_argument("--hair-dir", required=True, help="dir of hair-mask .npy files")
    parser.add_argument("--adapter", default=None, help="trained adapter dir (omit = stock)")
    parser.add_argument("--compare", action="store_true", help="also run stock + print delta")
    parser.add_argument("--limit", type=int, default=0, help="0 = all images")
    args = parser.parse_args()

    run(args.images, args.annotations, args.hair_dir, args.adapter, args.compare, args.limit)
