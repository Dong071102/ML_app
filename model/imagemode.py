import numpy as np
from diffusers import AutoPipelineForText2Image
import torch
from PIL import Image

pipeline = AutoPipelineForText2Image.from_pretrained(
                "warp-ai/wuerstchen", torch_dtype=torch.float32
            )
# load lora weights from folder:
pipeline.prior_pipe.load_lora_weights("dongOi071102/wuerstchen-prior-meme-lora-4", torch_dtype=torch.float32)
prompt='spider man see spider man'
image = pipeline(prompt=prompt).images[0]
print('type image ',type(image))

image.save("my_image.png")