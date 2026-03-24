@echo off
REM Flow-GRPO UV 자동 설정 스크립트 (Windows)
REM 사용법: setup_uv.bat [reward_type]
REM 예시: setup_uv.bat ocr

setlocal enabledelayedexpansion

echo ===================================
echo Flow-GRPO UV 환경 설정 (Windows)
echo ===================================

REM UV 설치 확인
uv --version >nul 2>&1
if errorlevel 1 (
    echo ❌ UV가 설치되지 않았습니다.
    echo 설치: https://docs.astral.sh/uv/getting-started/installation/
    exit /b 1
)

for /f "tokens=*" %%i in ('uv --version') do set UV_VERSION=%%i
echo ✓ UV 버전: %UV_VERSION%

REM Reward 타입 결정
set REWARD_TYPE=%1
if "%REWARD_TYPE%"=="" set REWARD_TYPE=base

echo.
if "%REWARD_TYPE%"=="base" (
    echo 📦 기본 의존성 설치중...
    call uv sync --no-dev
) else if "%REWARD_TYPE%"=="dev" (
    echo 🔧 개발 도구 포함 설치중...
    call uv sync --group dev
) else if "%REWARD_TYPE%"=="ocr" (
    echo 📝 OCR (Text Rendering) 설정중...
    call uv sync --extra ocr
) else if "%REWARD_TYPE%"=="geneval" (
    echo 🎯 GenEval Reward 설정중...
    call uv sync --extra geneval
) else if "%REWARD_TYPE%"=="imagereward" (
    echo 🖼️  ImageReward 설정중...
    call uv sync --extra imagereward
) else if "%REWARD_TYPE%"=="all-rewards" (
    echo 🏆 모든 Reward 모델 설정중...
    call uv sync --extra all-rewards
) else if "%REWARD_TYPE%"=="bagel" (
    echo 🚀 Bagel + Flash-Attn 설정중...
    call uv sync --extra bagel
) else (
    echo ❌ 알 수 없는 타입: %REWARD_TYPE%
    echo.
    echo 사용 가능한 옵션:
    echo   setup_uv.bat base          # 기본 설정 (PickScore 추천)
    echo   setup_uv.bat dev           # 개발 도구 포함
    echo   setup_uv.bat ocr           # OCR (텍스트 렌더링)
    echo   setup_uv.bat geneval       # GenEval Reward
    echo   setup_uv.bat imagereward   # ImageReward
    echo   setup_uv.bat all-rewards   # 모든 Reward 모델
    echo   setup_uv.bat bagel         # Bagel 모델
    exit /b 1
)

echo.
echo ===================================
echo ✅ 설정 완료!
echo ===================================
echo.
echo 다음 명령어로 환경 활성화:
echo   .venv\Scripts\Activate.ps1
echo.
echo 또는 UV로 직접 실행:
echo   uv run python scripts/train_sd3.py
echo.
echo 자세한 가이드: type UV_SETUP_KO.md
