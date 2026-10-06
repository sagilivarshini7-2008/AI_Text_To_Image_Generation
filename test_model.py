from diffusers import StableDiffusionPipeline
import torch

print("Loading model...")

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
)

pipe = pipe.to("cpu")

print("Model loaded successfully!")