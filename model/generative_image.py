import numpy as np
from diffusers import AutoPipelineForText2Image
import torch
from PIL import Image
import time
class TextToImageGenerator:
    def __init__(self, model_name="warp-ai/wuerstchen",lora_weights="dongOi071102/wuerstchen-prior-meme-lora-4" ):
        self.device = torch.device("mps") if torch.backends.mps.is_built() else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the text-to-image pipeline with float32 dtype
        self.model_name = model_name
        self.lora_weights = lora_weights
        self.pipeline = None
        print('load model')
    def load_model(self):
        self.pipeline = AutoPipelineForText2Image.from_pretrained(self.model_name, torch_dtype=torch.float32).to(self.device)
        
        # Load LoRA weights with float32 dtype (assuming "prior_pipe" is a member)
        self.pipeline.prior_pipe.load_lora_weights(self.lora_weights, torch_dtype=torch.float32)
        return "Model Loaded"
    def generate_image(self, prompt):
        # Generate the image
        image = self.pipeline(prompt=prompt).images[0]
        time.sleep(5)
        return image

    def show_image(self, prompt):
        # Generate and display the image
        image = self.generate_image(prompt)
        image.show()
        
    def save_image(self, prompt, filename="generated_image.png"):
        # Generate and save the image
        image = self.generate_image(prompt)
        image.save(filename)
        print(f"Image saved as {filename}")

