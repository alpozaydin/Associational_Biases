import os

import torch
from diffusers import StableDiffusion3Pipeline  # type: ignore
from dotenv import load_dotenv
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("could not get hf token — set the HF_TOKEN env var")

HF_HOME = os.environ.get("HF_HOME")
TRANSFORMERS_CACHE = os.environ.get("TRANSFORMERS_CACHE")


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = get_device()


def init_stable_diffusion() -> StableDiffusion3Pipeline:
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        torch_dtype=torch.float16,
        token=HF_TOKEN,
    )
    pipe = pipe.to(DEVICE)

    return pipe


def init_llava(*, to_gpu=True):
    processor = LlavaNextProcessor.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        cache_dir=TRANSFORMERS_CACHE,
    )

    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        torch_dtype=torch.float16,
        token=HF_TOKEN,
        cache_dir=TRANSFORMERS_CACHE,
    )

    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id  # type: ignore

    processor.patch_size = model.config.vision_config.patch_size  # type: ignore

    processor.vision_feature_select_strategy = (  # type: ignore
        model.config.vision_feature_select_strategy
    )

    if to_gpu:
        model.to(DEVICE)  # type: ignore

    return (processor, model)
