import numpy as np
from diffusers import AutoPipelineForText2Image
import torch
from PIL import Image
import time
pipeline = AutoPipelineForText2Image.from_pretrained(
                "warp-ai/wuerstchen", torch_dtype=torch.float32
            )
# load lora weights from folder:
pipeline.prior_pipe.load_lora_weights("dongOi071102/wuerstchen-prior-meme-image-no-text-lora-v1", torch_dtype=torch.float32)
index=0
while True:
    prompt=input('Astronaut in a jungle, cold color palette, muted colors, detailed, 8k')
    image = pipeline(prompt=prompt).images[0]
    image_name=f"my_image_{index}.png"
    image.save(image_name)
