import torch
from PIL import Image
import numpy as np
from diffusers import StableDiffusion3Pipeline
from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob
import importlib
from torch import amp

model_id = "stabilityai/stable-diffusion-3.5-medium"
device = "cuda"
read_token = ""

pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.float16, token=read_token)
# pipe = pipe.to(device)
pipe.enable_sequential_cpu_offload()
prompt = 'A steaming cup of coffee'
generator = torch.Generator()
generator.manual_seed(42) 
noise_level_list = [0,0.5,0.6,0.7,0.8]

# for noise_level in noise_level_list:
#     images, _, _, _ = pipeline_with_logprob(pipe,prompt,num_inference_steps=10,guidance_scale=4.5,output_type="pt",height=512,width=512,generator=generator,noise_level=noise_level)
#     pil = Image.fromarray((images[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
#     pil.save(f'scripts/demo/forward_sde-noise_level{noise_level}.png') 
for noise_level in noise_level_list:
    # autocast를 사용하여 데이터 타입을 자동으로 맞춤
    with torch.no_grad(), amp.autocast("cuda"):
        outputs = pipeline_with_logprob(
            pipe, prompt, num_inference_steps=10, guidance_scale=4.5, 
            output_type="pt", height=512, width=512, 
            generator=generator, noise_level=noise_level
        )
    
    images = outputs[0]
    # 결과가 float16일 수 있으므로 안전하게 float32로 변환 후 numpy 처리
    img_np = (images[0].detach().float().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    pil = Image.fromarray(img_np)
    pil.save(f'scripts/demo/forward_sde-noise_level{noise_level}.png')