from PIL import Image

from utils.initialisation import DEVICE


def describe_image_llava(
    processor,
    model,
    image: Image.Image,
    prompt="What is shown in this image? Focus on emotions.",
):
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
                {"type": "image"},
            ],
        },
    ]
    templated_prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True
    )

    inputs = processor(images=image, text=templated_prompt, return_tensors="pt").to(
        DEVICE
    )

    output = model.generate(**inputs, max_new_tokens=77)

    description = processor.decode(output[0], skip_special_tokens=True)

    return description.split("[/INST]")[1].strip()  # clean description and return
