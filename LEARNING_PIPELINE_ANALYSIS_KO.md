# flow_grpo 학습 파이프라인 상세 분석

## 목차
1. [학습 스크립트 구조](#1-학습-스크립트-구조)
2. [전체 파이프라인 흐름](#2-전체-파이프라인-흐름)
3. [모델 로드 및 설정](#3-모델-로드-및-설정)
4. [샘플링 (이미지 생성 & 로그확률 계산)](#4-샘플링-이미지-생성--로그확률-계산)
5. [보상 신호 계산](#5-보상-신호-계산)
6. [Advantage 계산](#6-advantage-계산)
7. [Policy 최적화](#7-policy-최적화)
8. [모델 저장](#8-모델-저장)

---

## 1. 학습 스크립트 구조

### 1.1 주요 스크립트 비교

| 파일 | 알고리즘 | 목적 |
|------|---------|------|
| `train_sd3.py` | **GRPO** | Timestep-level 정책 최적화 |
| `train_sd3_dpo.py` | **DPO** | Pair-wise (chosen/rejected) 비교 |
| `train_sd3_fast.py` | GRPO (축약) | 빠른 학습 버전 |

### 1.2 설정 파일 (config/)

```
base.py          → 기본 설정
grpo.py          → GRPO 특화 설정 (OCR, aesthetic 등)
dpo.py           → DPO 특화 설정 (geneval, pickscore 등)
sft.py           → Supervised Fine-Tuning 설정
```

**핵심 설정 예시 (config/grpo.py의 general_ocr_sd3)**
```python
config.sample.num_steps = 10              # 생성 시 디노이징 step
config.sample.num_image_per_prompt = 24   # 각 프롬프트당 샘플 수
config.train.timestep_fraction = 0.99     # 학습할 timestep 비율
config.train.beta = 0.0                   # KL loss 가중치
config.reward_fn = {"video_ocr": 1.0}     # 보상 함수
```

---

## 2. 전체 파이프라인 흐름

### 2.1 Epoch 단위 처리

```
Epoch 루프:
├─ [EVAL & CHECKPOINT] (매 config.eval_freq/save_freq 에포크)
├─ [SAMPLING] 이미지 생성 및 보상 계산
│  └─ 각 배치마다:
│     ├─ 프롬프트 텍스트 임베딩
│     ├─ pipeline_with_logprob()로 이미지 + 로그확률 생성
│     └─ reward_fn 비동기 호출 (Thread pool)
├─ [ADVANTAGE 계산] Per-prompt 통계 기반
├─ [TRAINING] config.train.num_inner_epochs 만큼 반복
│  └─ 각 timestep t에 대해:
│     ├─ 현재 모델 pred: compute_log_prob()
│     ├─ 참조 모델 pred: transformer.ref (DPO만 해당)
│     ├─ Loss 계산: policy_loss + beta*kl_loss
│     └─ Backward & optimizer.step()
└─ epoch += 1
```

---

## 3. 모델 로드 및 설정

### 3.1 기본 모델 로드 (train_sd3.py:397)

```python
pipeline = StableDiffusion3Pipeline.from_pretrained(
    config.pretrained.model  # e.g., "stabilityai/stable-diffusion-3.5-medium"
)

# 가중치 freezing
pipeline.vae.requires_grad_(False)
pipeline.text_encoder.requires_grad_(False)
pipeline.text_encoder_2.requires_grad_(False)
pipeline.text_encoder_3.requires_grad_(False)
pipeline.transformer.requires_grad_(not config.use_lora)
```

### 3.2 LoRA 설정 (train_sd3.py:418-433)

**GRPO 방식:**
```python
if config.use_lora:
    target_modules = [
        "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
        "attn.to_k", "attn.to_out.0", "attn.to_q", "attn.to_v"
    ]
    transformer_lora_config = LoraConfig(r=32, lora_alpha=64, init_lora_weights="gaussian")
    pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)
```

**DPO 방식 (train_sd3_dpo.py:422-432):**
```python
# learner와 ref 두 개의 LoRA adapter 생성
pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config, adapter_name="learner")
pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config, adapter_name="ref")
pipeline.transformer.set_adapter("learner")

# 학습 시 "learner" 파라미터만 수집
transformer_trainable_parameters = []
for name, param in transformer.named_parameters():
    if "learner" in name:
        assert param.requires_grad == True
        transformer_trainable_parameters.append(param)
```

### 3.3 EMA 모델 (train_sd3.py:436-437)

```python
ema = EMAModuleWrapper(
    transformer_trainable_parameters,
    decay=0.9,                    # EMA decay 계수
    update_step_interval=8,       # 8 step마다 EMA 업데이트
    device=accelerator.device
)
```

---

## 4. 샘플링 (이미지 생성 & 로그확률 계산)

### 4.1 GRPO 샘플링 (train_sd3.py:635-683)

**코드 흐름:**
```python
# 이미지 생성 + 각 timestep의 latent & log_prob 수집
with torch.no_grad():
    images, latents, log_probs = pipeline_with_logprob(
        pipeline,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_prompt_embeds=sample_neg_prompt_embeds,
        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
        num_inference_steps=config.sample.num_steps,
        guidance_scale=config.sample.guidance_scale,
        output_type="pt",
        height=config.resolution,
        width=config.resolution,
        noise_level=config.sample.noise_level,
    )

# 다차원 텐서 스택
latents = torch.stack(latents, dim=1)      # (batch, num_steps+1, 16, 96, 96)
log_probs = torch.stack(log_probs, dim=1)  # (batch, num_steps)
timesteps = pipeline.scheduler.timesteps.repeat(batch_size, 1)
```

**프로세스:**
- **latents**: [현재 latent] → [다음 latent] (각 timestep 추적)
- **log_probs**: 각 SDE step의 로그확률 (정책 그래디언트 학습에 필수)
- **images**: 최종 생성된 이미지 (VAE 디코딩)

### 4.2 DPO 샘플링 (train_sd3_dpo.py:654-680)

```python
if global_step > 0 and global_step % config.train.ref_update_step == 0:
    copy_learner_to_ref(transformer)  # ref 모델 업데이트

with torch.no_grad():
    pipeline.transformer.set_adapter("ref")  # Reference 모델 사용
    images, latents, log_probs = pipeline_with_logprob(...)

# DPO는 마지막 latent만 저장
samples.append({
    "latents": latents[-1],  # 최종 latent만
    ...
})
```

### 4.3 pipeline_with_logprob() 내부 동작 (sd3_pipeline_with_logprob.py)

```python
all_latents = [latents]  # 초기 숨겨진 noise
all_log_probs = []

for i, t in enumerate(timesteps):
    # Classifier-free guidance
    latent_model_input = torch.cat([latents] * 2)
    timestep = t.expand(latent_model_input.shape[0])
    
    # Transformer 예측
    noise_pred = self.transformer(
        hidden_states=latent_model_input,
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled_prompt_embeds,
    )[0]
    
    # Guidance 적용
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = (
        noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    )
    
    # SDE Step with log probability
    prev_sample, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
        scheduler,
        noise_pred,
        t,
        latents,
        prev_sample=next_latents,
        noise_level=noise_level,
    )
    
    all_latents.append(prev_sample)
    all_log_probs.append(log_prob)
    latents = prev_sample

return images, all_latents, all_log_probs
```

**SDE Step 내부 (sd3_sde_with_logprob.py):**
```python
def sde_step_with_logprob(...):
    # Flow Matching: zt = (1-t)*x + t*z1
    sigmas = scheduler.sigmas[step_indices]
    
    # 예측된 noise로부터 다음 latent 계산
    prev_sample = (zt - sigma_t * noise_pred) / (1 - sigma_t)
    
    # Gaussian log probability 계산
    # log p(z_{t-1} | z_t) = -0.5 * ||noise_pred - (z_{t-1} - (1-sigma)*z_t)||^2 / sigma^2
    log_prob = -0.5 * mean_squared_error / (sigma_t ** 2)
    
    return prev_sample, log_prob, prev_sample_mean, std_dev_t
```

---

## 5. 보상 신호 계산

### 5.1 Reward 함수 (flow_grpo/rewards.py)

**주요 Scorer 종류:**

| Scorer | 입력 | 출력 | 용도 |
|--------|------|------|------|
| `aesthetic_score()` | 이미지 | [0, 8] | 미적 품질 |
| `pickscore_score()` | 이미지, 프롬프트 | [0, 1] | 프롬프트-이미지 정렬 |
| `ocr_score()` | 이미지, 프롬프트 | [0, 1] | OCR 텍스트 인식 |
| `clip_score()` | 이미지, 프롬프트 | [0, 1] | CLIP 유사도 |
| `imagereward_score()` | 이미지, 프롬프트 | [-1, 1] | ImageReward 모델 |
| `qwenvl_score()` | 이미지, 프롬프트 | 실수 | QwenVL 평가 |
| `geneval_score()` | 이미지, 메타데이터 | dict | 복합 평가 (strict, group 등) |
| `video_ocr_score()` | 비디오/이미지, 프롬프트 | 실수 | 비디오 OCR |
| `jpeg_compressibility()` | 이미지 | 음수 | JPEG 압축률 (낮을수록 높음) |

### 5.2 Multi-Scorer 조합 (multi_score())

```python
# config/grpo.py 예시
config.reward_fn = {
    "ocr": 1.0,
    "aesthetic": 0.5,
    "pickscore": 0.2,
}

# rewards.py의 multi_score() 처리
def multi_score(device, score_dict):
    score_functions = {...}  # 모든 scorer 정의
    
    def _fn(images, prompts, metadata, only_strict=True):
        total_scores = []
        score_details = {}
        
        for score_name, weight in score_dict.items():
            scores, rewards = score_fns[score_name](images, prompts, metadata)
            score_details[score_name] = scores
            weighted_scores = [weight * score for score in scores]
            
            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [t + w for t, w in zip(total_scores, weighted_scores)]
        
        score_details['avg'] = total_scores
        return score_details, {}
    
    return _fn
```

### 5.3 비동기 보상 계산 (train_sd3.py:668-671)

```python
# 데이터 생성과 병렬로 보상 계산
rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
time.sleep(0)  # CPU에 제어권 양보

# 나중에 결과 수집
for sample in samples:
    rewards, reward_metadata = sample["rewards"].result()
    sample["rewards"] = {
        key: torch.as_tensor(value, device=accelerator.device).float()
        for key, value in rewards.items()
    }
```

---

## 6. Advantage 계산

### 6.1 Per-Prompt Advantage (train_sd3.py:760-783)

**PerPromptStatTracker (stat_tracking.py) 로직:**

```python
class PerPromptStatTracker:
    def update(self, prompts, rewards, type='grpo'):
        # 각 프롬프트별로 지금까지 수집된 모든 보상 저장
        for prompt in unique_prompts:
            self.stats[prompt] = [r1, r2, r3, ...]  # 발생한 모든 샘플
        
        # Advantage 계산
        for prompt in unique_prompts:
            prompt_rewards = rewards[prompts == prompt]
            mean = mean(self.stats[prompt])  # 평균
            std = std(self.stats[prompt]) + 1e-4  # 표준편차
            
            if type == 'grpo':
                advantages[prompts == prompt] = (prompt_rewards - mean) / std
            elif type == 'dpo':
                # chosen (max) → +1, rejected (min) → -1
                advantages[prompts == prompt] = sign(prompt_rewards)
```

**예시:**
```
프롬프트 A에 대해:
- 샘플 1: 보상 0.8
- 샘플 2: 보상 0.5
- 샘플 3: 보상 0.7

가우시안 정규화:
- mean = (0.8 + 0.5 + 0.7) / 3 = 0.667
- std = sqrt(var) ≈ 0.121
- advantages = (0.8 - 0.667) / 0.121 = +1.075
            = (0.5 - 0.667) / 0.121 = -1.377
            = (0.7 - 0.667) / 0.121 = +0.272
```

### 6.2 Advantage 필터링 (train_sd3.py:795-817)

```python
# Timestep 전체에서 advantage가 모두 0인 샘플 제거
mask = (samples["advantages"].abs().sum(dim=1) != 0)

# 배치 크기가 num_batches_per_epoch의 배수가 되도록 조정
num_batches = config.sample.num_batches_per_epoch
true_count = mask.sum()
if true_count % num_batches != 0:
    # False → True로 변경할 샘플 수
    num_to_change = num_batches - (true_count % num_batches)
    mask[false_indices[random_selection]] = True

# 필터링
samples = {k: v[mask] for k, v in samples.items()}
```

### 6.3 DPO의 별도 처리 (train_sd3_dpo.py:796-850)

DPO는 pair-wise 비교이므로 별도로 구성:
```python
# 고유 프롬프트별로 latent 그룹화
for prompt_id in unique_prompt_ids:
    matches = np.where(prompt_ids == prompt_id)[0]
    concat_latents.append(latents[matches])  # 같은 프롬프트의 모든 샘플
    concat_advantages.append(advantages[matches])

# [num_prompts, 2, ...] 형태로 정렬
# latents[i, 0] = chosen (reward 높음)
# latents[i, 1] = rejected (reward 낮음)
```

---

## 7. Policy 최적화

### 7.1 GRPO Loss 계산 (train_sd3.py:866-916)

**핵심 코드:**
```python
for j in train_timesteps:  # Timestep t = 0, 1, ..., num_train_timesteps-1
    with accelerator.accumulate(transformer):
        with autocast():
            # 현재 모델의 log prob 계산
            prev_sample, log_prob, prev_sample_mean, std_dev_t = compute_log_prob(
                transformer, pipeline, sample, j, embeds, pooled_embeds, config
            )
            
            # 참조 모델의 log prob 계산 (KL 손실 계산용)
            if config.train.beta > 0:
                with torch.no_grad():
                    with transformer.module.disable_adapter():  # 기본 모델 사용
                        _, _, prev_sample_mean_ref, _ = compute_log_prob(...)

        # GRPO 정책 손실
        advantages = torch.clamp(
            sample["advantages"][:, j],
            -config.train.adv_clip_max,  # [-5, +5] 클리핑
            config.train.adv_clip_max,
        )
        
        # 중요도 샘플링 비율
        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
        
        # PPO-스타일 loss (clipping)
        unclipped_loss = -advantages * ratio
        clipped_loss = -advantages * torch.clamp(
            ratio,
            1.0 - config.train.clip_range,  # [1e-4, 1.0001]
            1.0 + config.train.clip_range,
        )
        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
        
        # KL 발산 손실 (선택사항)
        if config.train.beta > 0:
            kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1,2,3), keepdim=True) / (2 * std_dev_t ** 2)
            kl_loss = torch.mean(kl_loss)
            loss = policy_loss + config.train.beta * kl_loss
        else:
            loss = policy_loss

        # 통계 수집
        info["policy_loss"].append(policy_loss)
        info["clipfrac"].append(torch.mean((torch.abs(ratio - 1.0) > config.train.clip_range).float()))
        info["loss"].append(loss)

        # 역전파
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(transformer.parameters(), config.train.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
```

**compute_log_prob() (train_sd3.py:238-268):**
```python
def compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config):
    # Classifier-free guidance 적용
    if config.train.cfg:
        noise_pred = transformer(
            hidden_states=torch.cat([sample["latents"][:, j]] * 2),
            timestep=torch.cat([sample["timesteps"][:, j]] * 2),
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
        )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = (
            noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        )
    else:
        noise_pred = transformer(...)
    
    # SDE step with log probability
    prev_sample, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
        pipeline.scheduler,
        noise_pred.float(),
        sample["timesteps"][:, j],
        sample["latents"][:, j].float(),
        prev_sample=sample["next_latents"][:, j].float(),
        noise_level=config.sample.noise_level,
    )
    
    return prev_sample, log_prob, prev_sample_mean, std_dev_t
```

### 7.2 DPO Loss 계산 (train_sd3_dpo.py:925-970)

**Diffusion-DPO 손실:**
```python
for j in train_timesteps:
    with accelerator.accumulate(transformer):
        # 모델 입력: [chosen, chosen, rejected, rejected]
        # (같은 noise로 paired samples 생성)
        bsz = model_input.shape[0] // 2
        noise = torch.randn_like(model_input)
        noise = torch.cat([noise[:bsz], noise[:bsz]], dim=0)  # 쌍 복제

        # Learner 모델 예측
        pipeline.transformer.set_adapter("learner")
        model_pred = transformer(
            hidden_states=noisy_model_input,
            timestep=timesteps,
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
        )[0]

        # Reference 모델 예측
        with torch.no_grad():
            pipeline.transformer.set_adapter("ref")
            model_pred_ref = transformer(...)[0]
            pipeline.transformer.set_adapter("learner")

        target = noise - model_input

        # MSE 기반 DPO 손실
        theta_mse = ((model_pred - target) ** 2).reshape(target.shape[0], -1).mean(dim=1)
        ref_mse = ((model_pred_ref - target) ** 2).reshape(target.shape[0], -1).mean(dim=1)

        # Split: chosen/rejected
        model_w_err = theta_mse[:bsz]  # Chosen
        model_l_err = theta_mse[bsz:]  # Rejected
        ref_w_err = ref_mse[:bsz]
        ref_l_err = ref_mse[bsz:]

        # DPO divergence
        w_diff = model_w_err - ref_w_err
        l_diff = model_l_err - ref_l_err
        w_l_diff = w_diff - l_diff  # Chosen MSE - Rejected MSE

        # Binary cross-entropy 형태의 손실
        inside_term = -0.5 * config.train.beta * w_l_diff
        loss = -F.logsigmoid(inside_term)  # Binary classification loss
        loss = torch.mean(loss)

        info["loss"].append(loss)
        info["w_l_diff"].append(torch.mean(w_l_diff))
        
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(transformer.parameters(), config.train.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
```

### 7.3 Gradient Accumulation

```python
# config/base.py
train.gradient_accumulation_steps = 2  # 배치 크기 효율화

# 실제 설정
accelerator = Accelerator(
    gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    # num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)
    # 예: 2 * 10 = 20 step마다 optimizer.step() 호출
)

for j in train_timesteps:
    with accelerator.accumulate(transformer):
        # Gradient 축적
        accelerator.backward(loss)
        
    # accelerator 내부에서 20 step마다 자동으로 optimizer.step() 호출
    if accelerator.sync_gradients:
        # 동기화된 그래디언트 → optimizer.step() 실행됨
        global_step += 1
```

### 7.4 EMA 모델 업데이트 (train_sd3.py:923-924)

```python
if config.train.ema:
    ema.step(transformer_trainable_parameters, global_step)
    # decay 계수: min((1 + global_step) / (10 + global_step), 0.9)
    # EMA: ema_param = decay * ema_param + (1 - decay) * current_param
```

---

## 8. 모델 저장

### 8.1 Checkpoint 저장 (train_sd3.py:308-320)

```python
def save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config):
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)
    
    if accelerator.is_main_process:
        # EMA 모델을 현재 파라미터에 복사
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        
        # LoRA 가중치만 저장
        unwrap_model(transformer, accelerator).save_pretrained(save_root_lora)
        
        # EMA 복사본 복원
        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)
```

### 8.2 저장된 구조

```
logs/ocr/sd3.5-M-grpo/
├── checkpoints/
│   ├── checkpoint-100/
│   │   └── lora/
│   │       ├── adapter_config.json
│   │       └── adapter_model.bin          # LoRA 가중치
│   ├── checkpoint-200/
│   └── checkpoint-300/
├── train_config_image_0.json
└── logs.txt
```

### 8.3 체크포인트 관리

```python
accelerator_config = ProjectConfiguration(
    project_dir=os.path.join(config.logdir, config.run_name),
    automatic_checkpoint_naming=True,
    total_limit=config.num_checkpoint_limit,  # 최대 5개 유지
)
# 자동으로 오래된 체크포인트 삭제
```

---

## 9. 주요 차이점: GRPO vs DPO

### 9.1 샘플링 단계

| 항목 | GRPO | DPO |
|------|------|-----|
| 저장 정보 | 모든 timestep의 latent & log_prob | 최종 latent만 |
| 보상 군집 | 같은 프롬프트 모두 저장 | Pair (chosen/rejected) |
| 파라미터 | Single adapter | learner + ref adapters |

### 9.2 최적화 단계

| 항목 | GRPO | DPO |
|------|------|-----|
| Loss 타입 | PPO-style clipped policy | Binary cross-entropy |
| Timestep 루프 | 있음 (각 t마다 grad update) | 있음 (하지만 pair 처리) |
| Ref 모델 업데이트 | Disable adapter로 기본 모델 사용 | copy_learner_to_ref() |
| 추적 메트릭 | approx_kl, clipfrac, policy_loss | model_w_err, model_l_err |

---

## 10. 데이터 흐름 예시

### 10.1 GRPO 전체 흐름 예시

```
Input Prompts: ["A cat sitting", "A dog running"]
num_image_per_prompt: 2

Step 1: SAMPLING
├─ Prompt 0: "A cat sitting"
│  ├─ Image 0: cat sitting (reward=0.8)
│  ├─ Image 1: cat sleeping (reward=0.6)
│  ├─ latents: [noise, z₁, z₂, ..., z₁₀]
│  └─ log_probs: [log_p₀, log_p₁, ..., log_p₉]
└─ Prompt 1: "A dog running"
   ├─ Image 0: dog running (reward=0.9)
   ├─ Image 1: dog lying (reward=0.4)
   └─ latents, log_probs: ...

Step 2: ADVANTAGE CALCULATION
├─ Prompt 0: mean=0.7, std=0.1
│  ├─ Image 0 adv: (0.8-0.7)/0.1 = +1.0
│  └─ Image 1 adv: (0.6-0.7)/0.1 = -1.0
└─ Prompt 1: mean=0.65, std=0.25
   ├─ Image 0 adv: (0.9-0.65)/0.25 = +1.0
   └─ Image 1 adv: (0.4-0.65)/0.25 = -1.0

Step 3: TRAINING (for each timestep t)
├─ t=0:
│  ├─ compute_log_prob() → log_p'₀ (현재 모델)
│  ├─ ratio = exp(log_p'₀ - log_p₀) (원래 샘플링 시 log_prob)
│  ├─ loss = max(-adv * ratio, -adv * clipped_ratio)
│  └─ backward() & step()
├─ t=1: ...
└─ t=9: ...

Step 4: CHECKPOINT SAVE
└─ Save: lora/adapter_model.bin
```

### 10.2 DPO 전체 흐름 예시

```
Input Prompts: ["A cat sitting", "A dog running"]

Step 1: SAMPLING (with ref model)
├─ Generate images using ref model
├─ Store final latents only
└─ Rewards: cat_sitting→[0.8, 0.6], dog_running→[0.9, 0.4]

Step 2: ADVANTAGE CALCULATION (DPO mode)
├─ Prompt 0: max=0.8 (chosen), min=0.6 (rejected)
│  ├─ Image 0 adv: +1.0
│  └─ Image 1 adv: -1.0
└─ Prompt 1: max=0.9 (chosen), min=0.4 (rejected)
   ├─ Image 0 adv: +1.0
   └─ Image 1 adv: -1.0

Step 3: TRAINING (pair-wise)
├─ Input: [chosen₀, rejected₀, chosen₁, rejected₁] (same noise)
├─ For j in timesteps:
│  ├─ model_pred_learner (4 samples)
│  ├─ model_pred_ref (4 samples, no grad)
│  ├─ MSE分割: [chosen_mse, rejected_mse]
│  ├─ loss = sigmoid(-(chosen_mse - ref_chosen_mse) + (rejected_mse - ref_rejected_mse))
│  └─ backward() & step()

Step 4: CHECKPOINT SAVE
├─ Ref 모델 주기적 업데이트
└─ Save: lora/adapter_model.bin (learner만)
```

---

## 11. 성능 최적화

### 11.1 비동기 보상 계산
```python
# 이미지 생성 중에 보상 계산 시작
rewards = executor.submit(reward_fn, images, prompts, metadata)
# 다음 배치 샘플링 중에 보상 계산 진행
# 마지막에 result() 호출해서 결과 수집
```

### 11.2 혼합 정밀도 학습
```python
config.mixed_precision = "bf16"  # bfloat16 사용
# VAE, text_encoder: fp32 (inference only)
# transformer: bf16 (학습)
```

### 11.3 Gradient Accumulation
```python
# 메모리 절감을 위해 여러 step의 gradient 축적
config.train.gradient_accumulation_steps = 2
# 실제로는 2 * num_train_timesteps = 20 step마다 update
```

### 11.4 LoRA 사용
```python
# 전체 모델의 0.1% 정도의 파라미터만 학습
# r=32, alpha=64 → 약 1-2M 파라미터만 학습
# full-parameter tuning 대비 10배 이상 메모리 절감
```

---

## 12. 주요 하이퍼파라미터

```python
# Sampling
config.sample.num_steps = 40              # 디노이징 step
config.sample.num_image_per_prompt = 24   # 각 프롬프트당 샘플
config.sample.guidance_scale = 4.5        # CFG 강도
config.sample.noise_level = 0.7           # SDE 노이즈 레벨

# Training
config.train.timestep_fraction = 0.99     # 학습할 step 비율
config.train.beta = 0.0 or 100            # KL loss (GRPO) / DPO beta
config.train.learning_rate = 1e-4         # 학습률
config.train.adv_clip_max = 5             # Advantage clipping
config.train.clip_range = 1e-4            # PPO clip range
config.train.max_grad_norm = 1.0          # Gradient clipping

# LoRA
r = 32, alpha = 64

# EMA
decay = 0.9, update_interval = 8
```

---

## 결론

flow_grpo 프로젝트는 **확산 모델 정책 최적화**를 위한 두 가지 주요 알고리즘을 구현합니다:

1. **GRPO**: Timestep별 정책 그래디언트 (PPO-style)
2. **DPO**: Pair-wise 비교 (Diffusion DPO)

핵심 혁신:
- SDE step에서 정확한 log probability 계산
- Per-prompt advantage 정규화로 분산 감소
- 비동기 보상 계산으로 처리 가속
- LoRA + EMA로 효율적 학습

