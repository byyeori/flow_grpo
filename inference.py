import torch
from diffusers import StableDiffusion3Pipeline
from peft import PeftModel

device = "cuda"

# base 모델
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    torch_dtype=torch.float16
).to(device)

# LoRA 로드 (중요)
pipe.transformer = PeftModel.from_pretrained(
    pipe.transformer,
    "checkpoints/checkpoint-XXXX/lora"
)

pipe.transformer.eval()

# prompt
prompt = "a photo of a cat sitting on a table"

# inference
image = pipe(
    prompt=prompt,
    num_inference_steps=8,
    guidance_scale=3.5,
).images[0]

image.save("result.png")