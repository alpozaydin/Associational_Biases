"""Frozen LLaVA-NeXT describer with a swappable LoRA adapter.

This is the plug-and-play artifact of the project: a LoRA adapter that drops
onto a stock, otherwise-frozen ``llava-hf/llava-v1.6-mistral-7b-hf`` backbone.
No backbone weights are trained; only the adapter is.

Placement presets escalate per Sec 4.3 of the roadmap (resolve the RQ-B risk
empirically, do not assume):
    - ``llm_only``        : LoRA on the Mistral language tower only.
    - ``llm_projector``   : + the multi-modal projector.
    - ``llm_projector_vision_late`` : + the late CLIP vision-tower layers.

Mirrors the model/processor setup in
``inter-model communication pipeline/utils/initialisation.py`` (kept separate
because that directory's name has a space and is not importable as a package).
"""

import os

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

MODEL_PATH = "llava-hf/llava-v1.6-mistral-7b-hf"

# CLIP ViT-L/14 in llava-v1.6 has 24 encoder layers (indices 0-23). "Late" =
# the last block, matching the Grad-CAM probe region (vision-tower layer 9 is
# read by the explainability pipeline; we adapt layers above it when escalating).
_VISION_LATE_LAYERS = tuple(range(18, 24))

# Attention + MLP projection module name suffixes per sub-network.
_LLM_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
_VISION_TARGETS = ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2")


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = get_device()


def _target_modules_regex(placement: str) -> str:
    """Build a full-match regex selecting the modules LoRA wraps.

    peft treats a ``str`` ``target_modules`` as a regex matched against each
    module's fully-qualified name, so we can scope to a sub-network precisely
    (a plain suffix list would also catch the vision tower's ``q_proj``).
    """
    llm = rf"language_model\..*\.({'|'.join(_LLM_TARGETS)})"
    if placement == "llm_only":
        return rf"^.*(?:{llm})$"

    projector = r"multi_modal_projector\..*"
    if placement == "llm_projector":
        return rf"^.*(?:{llm}|{projector})$"

    if placement == "llm_projector_vision_late":
        layers = "|".join(str(i) for i in _VISION_LATE_LAYERS)
        vision = (
            rf"vision_tower\.vision_model\.encoder\.layers\.(?:{layers})"
            rf"\..*\.({'|'.join(_VISION_TARGETS)})"
        )
        return rf"^.*(?:{llm}|{projector}|{vision})$"

    raise ValueError(
        f"unknown placement {placement!r}; expected one of "
        "'llm_only', 'llm_projector', 'llm_projector_vision_late'"
    )


def _load_processor() -> LlavaNextProcessor:
    processor = LlavaNextProcessor.from_pretrained(MODEL_PATH)
    # Patches required for LlavaNext image-token expansion (see initialisation.py).
    base_cfg = LlavaNextForConditionalGeneration.config_class.from_pretrained(MODEL_PATH)
    processor.patch_size = base_cfg.vision_config.patch_size
    processor.vision_feature_select_strategy = base_cfg.vision_feature_select_strategy
    return processor


def _load_backbone(*, to_device: bool) -> LlavaNextForConditionalGeneration:
    hf_token = os.environ.get("HF_TOKEN")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        token=hf_token,
        cache_dir=os.environ.get("TRANSFORMERS_CACHE"),
    )
    if to_device:
        model.to(DEVICE)
    return model


def build_lora_describer(
    placement: str = "llm_only",
    *,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    to_device: bool = True,
):
    """Load a frozen LLaVA backbone with a fresh (zero-init) LoRA adapter.

    The returned model has all backbone weights frozen; only LoRA params
    require grad. A fresh adapter is identity (LoRA B is zero-initialised), so
    outputs match the stock model until the adapter is trained — run a "bite"
    tune before expecting Grad-CAM to move.

    :param placement: which sub-networks LoRA wraps (see module docstring).
    :returns: ``(processor, peft_model)``.
    """
    processor = _load_processor()
    base = _load_backbone(to_device=to_device)

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=_target_modules_regex(placement),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, config)
    model.print_trainable_parameters()
    return processor, model


def load_adapter_describer(adapter_path: str, *, to_device: bool = True):
    """Load a trained adapter onto a stock frozen backbone for plug-and-play eval.

    This is the deploy path: stock LLaVA + ``adapter_path``, no retraining.

    :param adapter_path: directory written by ``save_adapter`` / PEFT.
    :returns: ``(processor, peft_model)`` in eval mode.
    """
    processor = _load_processor()
    base = _load_backbone(to_device=to_device)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return processor, model


def save_adapter(model, path: str) -> None:
    """Persist only the LoRA adapter (the shippable artifact), not the backbone."""
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
