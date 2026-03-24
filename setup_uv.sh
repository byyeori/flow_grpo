#!/bin/bash

# Flow-GRPO UV 자동 설정 스크립트 (macOS/Linux)
# 사용법: bash setup_uv.sh [reward_type]
# 예시: bash setup_uv.sh ocr

set -e  # 에러 발생 시 중단

echo "==================================="
echo "Flow-GRPO UV 환경 설정"
echo "==================================="

# UV 설치 확인
if ! command -v uv &> /dev/null; then
    echo "❌ UV가 설치되지 않았습니다."
    echo "설치: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✓ UV 버전: $(uv --version)"

# Reward 타입 결정
REWARD_TYPE="${1:-base}"

case "$REWARD_TYPE" in
    base)
        echo "📦 기본 의존성 설치중..."
        uv sync --no-dev
        ;;
    dev)
        echo "🔧 개발 도구 포함 설치중..."
        uv sync --group dev
        ;;
    ocr)
        echo "📝 OCR (Text Rendering) 설정중..."
        uv sync --extra ocr
        ;;
    geneval)
        echo "🎯 GenEval Reward 설정중..."
        uv sync --extra geneval
        ;;
    imagereward)
        echo "🖼️  ImageReward 설정중..."
        uv sync --extra imagereward
        ;;
    all-rewards)
        echo "🏆 모든 Reward 모델 설정중..."
        uv sync --extra all-rewards
        ;;
    bagel)
        echo "🚀 Bagel + Flash-Attn 설정중..."
        uv sync --extra bagel
        ;;
    *)
        echo "❌ 알 수 없는 타입: $REWARD_TYPE"
        echo ""
        echo "사용 가능한 옵션:"
        echo "  bash setup_uv.sh base          # 기본 설정 (PickScore 추천)"
        echo "  bash setup_uv.sh dev           # 개발 도구 포함"
        echo "  bash setup_uv.sh ocr           # OCR (텍스트 렌더링)"
        echo "  bash setup_uv.sh geneval       # GenEval Reward"
        echo "  bash setup_uv.sh imagereward   # ImageReward"
        echo "  bash setup_uv.sh all-rewards   # 모든 Reward 모델"
        echo "  bash setup_uv.sh bagel         # Bagel 모델"
        exit 1
        ;;
esac

echo ""
echo "==================================="
echo "✅ 설정 완료!"
echo "==================================="
echo ""
echo "다음 명령어로 환경 활성화:"
echo "  source .venv/bin/activate"
echo ""
echo "또는 UV로 직접 실행:"
echo "  uv run python scripts/train_sd3.py"
echo ""
echo "자세한 가이드: cat UV_SETUP_KO.md"
