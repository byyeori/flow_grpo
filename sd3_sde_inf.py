import torch
from PIL import Image
import numpy as np
from diffusers import StableDiffusion3Pipeline
from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob
from peft import PeftModel
from torch import amp
model_id = "stabilityai/stable-diffusion-3.5-medium"
device = "cuda"

# v_θ(x, t) = f(x, t; W + ΔW_lora) 여기서 lora를 학습함
# base model = 일반적인 diffusion 방향
# lora = reward 방향으로 미세하게 방향 틀기

# 1. 파이프라인 로드
pipe = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)
# 2. 학습한 LoRA 적용 (실제 경로로 수정)
pipe.transformer = PeftModel.from_pretrained(
    pipe.transformer,
    "outputs/sd35_grpo_pickscore/checkpoints/checkpoint-80/lora"
)

# 메모리가 넉넉하다면 오프로드 대신 바로 CUDA로 올리는 게 빠릅니다.
pipe.to(device)

prompt = 'A steaming cup of coffee'
noise_level_list = [0, 0.5, 0.6, 0.7, 0.8]

for noise_level in noise_level_list:
    # 각 노이즈별로 동일한 시작 포인트를 갖기 위해 루프 안에서 제너레이터 설정
    generator = torch.Generator(device=device).manual_seed(42)
    
    with torch.no_grad(), amp.autocast("cuda"):
        outputs = pipeline_with_logprob(
            pipe,
            prompt,
            num_inference_steps=10, # SD 3.5는 20~30단계를 추천하지만 테스트용으론 OK
            guidance_scale=3.5,
            output_type="pt",
            height=512,
            width=512,
            generator=generator,
            noise_level=noise_level
        )

    images = outputs[0]
    # PT -> NumPy 변환 및 저장
    img_np = (images[0].detach().float().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    Image.fromarray(img_np).save(f'result_noise_{noise_level}.png')
    print(f"Saved: result_noise_{noise_level}.png")
