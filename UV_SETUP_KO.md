# UV 기반 Flow-GRPO 환경 설정 가이드

이 가이드는 `uv` 패키지 매니저를 사용하여 Flow-GRPO 프로젝트를 설정하는 방법입니다.

## 📦 UV란?

`uv`는 Rust로 작성된 극도로 빠른 Python 패키지 설치 도구입니다.
- pip보다 5-100배 빠름
- 프로젝트 및 스크립트 관리
- 모든 작업을 별도 가상 환경에서 수행

**설치**: https://docs.astral.sh/uv/getting-started/installation/

---

## 🚀 빠른 시작 (추천)

### 1단계: UV 설치

```bash
# macOS (Homebrew)
brew install uv

# Linux / Windows
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2단계: Flow-GRPO 환경 생성

```bash
cd /path/to/flow_grpo

# Python 3.10 가상 환경 자동 생성 및 의존성 설치
uv sync

# (선택) 개발 도구도 함께 설치
uv sync --group dev
```

### 3단계: 환경 활성화 (선택)

```bash
# PowerShell (Windows)
.venv\Scripts\Activate.ps1

# Bash/Zsh (macOS/Linux)
source .venv/bin/activate

# 또는 UV로 직접 실행 (환경 활성화 불필요)
uv run python train_script.py
```

---

## 🎯 Reward 모델별 설정

### PickScore (권장: 가장 간단함)

```bash
# 기본 의존성만 설치
uv sync
```

### GenEval

```bash
uv sync --extra geneval
```

### OCR (Text Rendering)

```bash
uv sync --extra ocr
```

### ImageReward

```bash
uv sync --extra imagereward
```

### 모든 Reward 모델

```bash
uv sync --extra all-rewards
```

### Bagel (Byte Dance) 모델

```bash
uv sync --extra bagel
```

### Flash-Attn (선택적 최적화)

```bash
uv sync --extra flash-attn
```

---

## 🏃 스크립트 실행

### 방법 1: 활성화된 환경에서 실행

```bash
source .venv/bin/activate
python scripts/train_sd3.py --config config/grpo.py:pickscore_sd3
```

### 방법 2: UV로 직접 실행 (권장)

```bash
# 가상 환경 활성화 불필요
uv run python scripts/train_sd3.py --config config/grpo.py:pickscore_sd3

# accelerate 사용
uv run accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml scripts/train_sd3.py
```

### 방법 3: 가상 환경 쉘 진입

```bash
uv run --no-sync bash
# 또는
uv venv --python 3.10
source .venv/bin/activate
```

---

## 📋 유용한 UV 명령어

```bash
# 추가 패키지 설치
uv pip install <package_name>

# 설치된 패키지 목록 확인
uv pip list

# 의존성 잠금 파일 생성 (선택)
uv lock

# 가상 환경 제거
rm -r .venv

# 가상 환경 경로 확인
uv venv --python 3.10 --show

# 특정 Python 버전으로 환경 생성
uv sync --python 3.11
```

---

## 🔄 Conda에서 UV로 마이그레이션

기존 conda 환경에서 이미 학습 중이었다면:

```bash
# 1. 기존 conda 환경 확인
conda env list

# 2. UV로 새 환경 생성 (conda와 병렬 운영 가능)
cd flow_grpo
uv sync

# 3. 기존 conda 환경은 유지 또는 삭제
conda remove --name flow_grpo --all  # 선택사항
```

---

## 🐛 트러블슈팅

### 문제 1: "uv command not found"

```bash
# UV 재설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# PATH 업데이트 후 터미널 재실행
```

### 문제 2: GPU 관련 에러

```bash
# PyTorch 버전 확인
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# CUDA 12.1 기반으로 재설치 필요한 경우
# pyproject.toml의 torch 버전 수정 후:
uv sync --refresh
```

### 문제 3: 특정 패키지 설치 실패

```bash
# 상세 로그 확인
uv sync -v

# 캐시 제거 후 재시도
uv cache clean
uv sync
```

### 문제 4: OOM (메모리 부족)

UV 자체와는 무관하지만, 학습 시:

```bash
# LoRA 모드로 메모리 절약
# config/grpo.py에서 설정 수정 후 실행
uv run python scripts/train_sd3.py --config config/grpo.py:pickscore_bagel_lora
```

---

## 📌 주의사항

1. **torch 버전**: CUDA 버전과 일치 확인 필수
   ```bash
   # CUDA 버전 확인
   nvcc --version
   ```

2. **Diffusers 업그레이드** (FLUX/Qwen-Image 사용 시):
   ```bash
   uv pip install --upgrade diffusers
   ```

3. **Reward 서버** (UnifiedReward 사용 시):
   ```bash
   # 별도 터미널에서 실행
   uv run python -m sglang.launch_server --model-path CodeGoat24/UnifiedReward-7b-v1.5 --api-key flowgrpo --port 17140
   ```

---

## ✅ 설정 확인

```bash
# 모든 의존성 설치 확인
uv run python -c "
import torch, transformers, diffusers, accelerate
print('✓ PyTorch:', torch.__version__)
print('✓ Transformers:', transformers.__version__)
print('✓ Diffusers:', diffusers.__version__)
print('✓ Accelerate:', accelerate.__version__)
print('✓ CUDA Available:', torch.cuda.is_available())
"
```

---

## 🎓 추가 자료

- [UV 공식 문서](https://docs.astral.sh/uv/)
- [UV vs pip 비교](https://docs.astral.sh/uv/pip/)
- [Flow-GRPO 원본 README](README.md)
