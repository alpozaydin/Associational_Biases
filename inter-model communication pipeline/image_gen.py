# source for image gen code: https://huggingface.co/stabilityai/stable-diffusion-2-1
import torch
from diffusers import (
    DPMSolverMultistepScheduler,  # type: ignore
    StableDiffusionPipeline,  # type: ignore
)

# loading models

model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

import platform

_device = "mps" if platform.system() == "Darwin" else ("cuda" if torch.cuda.is_available() else "cpu")
pipe = pipe.to(_device)

# Recommended if your computer has < 64 GB of RAM
pipe.enable_attention_slicing()


# loading prompts

# df = pd.read_csv('initial_prompts.csv')

# for prompt in tqdm(df.itertuples(), desc="Iterating through prompts...", total=len(df), leave=True, dynamic_ncols=True):
#     image = pipe(prompt.initial_prompt, num_inference_steps=50, width=512, height=512).images[0]
#     image.save(f"initial_images/{prompt.prompt_name}_50_512x512.png")


image = pipe(
    "a photo of a chemistry teacher in a school classroom",
    num_inference_steps=50,
    width=768,
    height=768,
).images[0]
image.save("chem_in_classroom.png")
