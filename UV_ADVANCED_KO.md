# UV로 Flow-GRPO 실행하기 - 추가 팁

## 🎯 가장 빠른 시작 (30초)

```bash
# 1. UV 설치 (처음 한 번만)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Flow-GRPO 클론 및 진입
git clone https://github.com/yifan123/flow_grpo.git
cd flow_grpo

# 3. 환경 생성 (1-2분)
bash setup_uv.sh base     # macOS/Linux
# 또는
setup_uv.bat base         # Windows

# 4. 학습 시작
uv run python scripts/train_sd3.py --config config/grpo.py:pickscore_sd3
```

---

## 📊 Conda vs UV 비교

| 항목 | Conda | UV |
|------|-------|-----|
| 설치 속도 | 느림 (5-10분) | 매우 빠름 (1-2분) |
| 메모리 사용 | 많음 | 적음 |
| 보안 업데이트 | 느림 | 자동 |
| 의존성 충돌 | 많음 | 적음 |
| Python 관리 | 복잡 | 자동 |
| 멀티 프로젝트 | 환경 분리 필요 | .venv로 격리됨 |

**결론**: UV를 먼저 시도하세요. Conda로 돌아가기는 언제든 가능합니다.

---

## 🔧 고급 사용법

### 1. 특정 Python 버전 사용

```bash
# Python 3.11로 환경 생성
uv sync --python 3.11

# Python 3.9 사용 (특정 라이브러리 호환성)
uv sync --python 3.9
```

### 2. 특정 패키지만 업그레이드

```bash
# 단일 패키지 업데이트
uv pip install --upgrade torch

# 여러 패키지 업데이트
uv pip install --upgrade torch diffusers transformers
```

### 3. 추가 의존성 임시 설치 (lock 파일 수정 없이)

```bash
# 임시로 debugpy 설치
uv run --with debugpy python -m debugpy.adapter

# 또는 직접 pip 사용
uv pip install debugpy
```

### 4. Lock 파일 명시적으로 생성

```bash
# 현재 의존성을 uv.lock에 저장
uv lock

# Git에 커밋하여 팀원과 동일 버전 공유
git add uv.lock
git commit -m "Lock dependencies"
```

### 5. 다른 팀원과 동일한 환경 공유

```bash
# 팀원 1: 작업 완료 후
uv lock
git add uv.lock
git push

# 팀원 2: 동일 환경으로 복원
git pull
uv sync  # uv.lock 기반으로 정확히 같은 버전 설치
```

---

## 🚀 멀티 GPU 학습 실행

### 방법 1: 활성화된 환경에서

```bash
# 환경 활성화
source .venv/bin/activate

# 4개 GPU로 학습 시작
accelerate launch --multi-gpu --num_processes=4 scripts/train_sd3.py \
  --config config/grpo.py:pickscore_sd3
```

### 방법 2: UV로 직접 (권장)

```bash
# 환경 활성화 없이 바로 실행
uv run accelerate launch --multi-gpu --num_processes=4 scripts/train_sd3.py \
  --config config/grpo.py:pickscore_sd3
```

### 방법 3: 멀티노드 학습

```bash
# 마스터 노드
uv run bash scripts/multi_node/sd3.sh 0

# 워커 노드 (다른 머신)
uv run bash scripts/multi_node/sd3.sh 1
uv run bash scripts/multi_node/sd3.sh 2
```

---

## 📝 스크립트 예제

### `run_training.sh` 예제

```bash
#!/bin/bash
# UV 기반 학습 스크립트

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# UV 환경 확인
if ! command -v uv &> /dev/null; then
    echo "UV 설치 필요"
    exit 1
fi

# 의존성 설치 (선택)
if [ ! -d ".venv" ]; then
    echo "환경 생성중..."
    bash setup_uv.sh ocr
fi

# 학습 시작
echo "학습 시작..."
uv run accelerate launch \
    --config_file scripts/accelerate_configs/multi_gpu.yaml \
    scripts/train_sd3.py \
    --config config/grpo.py:geneval_sd3

echo "학습 완료!"
```

사용법:
```bash
bash run_training.sh
```

---

## 🔍 문제 해결

### 문제 1: "python" command not found

```bash
# UV로 실행해야 함
uv run python --version
```

### 문제 2: accelerate 명령어 인식 안됨

```bash
# 해결책 1: 환경 활성화
source .venv/bin/activate
accelerate launch scripts/train_sd3.py

# 해결책 2: UV run 사용
uv run accelerate launch scripts/train_sd3.py
```

### 문제 3: CUDA 에러

```bash
# 설치된 PyTorch 확인
uv run python -c "import torch; print(torch.cuda.is_available())"

# 현재 설치된 패키지 확인
uv pip list | grep torch

# CUDA 12.1용으로 강제 재설치
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

### 문제 4: 메모리 부족

```bash
# LoRA 학습 (메모리 절약)
uv run python scripts/train_sd3.py --config config/grpo.py:pickscore_bagel_lora

# 배치 크기 축소
uv run python -c "
import sys
sys.path.insert(0, '.')
from config.grpo import *
general_config.sample.train_batch_size = 2  # 기본값 개수 감소
"
```

---

## ✨ UV 팁과 트릭

### 1. 여러 프로젝트 관리

```bash
# 프로젝트별로 자동으로 다른 .venv 생성
project1/
  .venv/           # project1 전용
  pyproject.toml

project2/
  .venv/           # project2 전용
  pyproject.toml
```

### 2. 환경 변수 관리

```bash
# .env 파일 존재 시 자동으로 로드
# .env 예제:
WANDB_API_KEY=your_key
HF_TOKEN=your_token

# UV run으로 자동 적용
uv run python scripts/train_sd3.py
```

### 3. 깔끔한 정리

```bash
# 불필요한 의존성 제거 (uv.lock 기반)
uv sync

# 캐시 정리
uv cache clean

# 환경 삭제 후 재생성
rm -rf .venv uv.lock
uv sync
```

### 4. 성능 모니터링

```bash
# 환경 경로 확인
uv venv --show

# 설치된 패키지 트리 보기
uv pip tree

# 의존성 업데이트 확인
uv pip outdated
```

---

## 📚 추가 자료

- [UV 공식 가이드](https://docs.astral.sh/uv/)
- [pyproject.toml 스펙](https://packaging.python.org/en/latest/specifications/declaring-project-metadata/)
- [Flow-GRPO 원본 README](README.md)
- [Accelerate 문서](https://huggingface.co/docs/accelerate/)

---

## 💬 FAQ

**Q: 기존 Conda 환경에서 데이터를 계속 사용할 수 있나요?**
A: 네, 데이터는 환경과 무관하게 저장되어 있으므로 UV 환경에서도 그대로 사용 가능합니다.

**Q: UV와 Conda를 함께 사용해도 되나요?**
A: 네, 하지만 같은 프로젝트에서는 하나만 사용하는 것을 권장합니다.

**Q: pyproject.toml을 수정할 때마다 재설치해야 하나요?**
A: `uv sync` 또는 `uv lock`으로 자동 감지 및 적용됩니다.

**Q: 특정 버전의 패키지를 고정하려면?**
A: pyproject.toml에서 버전을 명시하면 uv.lock에 자동으로 고정됩니다.

---

**Happy Training! 🚀**
