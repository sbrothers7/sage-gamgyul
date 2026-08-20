#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "============================================================"
echo "  감귤 병해 AI 연구 환경 설치"
echo "============================================================"
echo

PYTHON_BIN=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON_BIN="$candidate"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "  [오류] 파이썬이 설치되어 있지 않습니다."
    echo
    echo "  1. https://www.python.org/downloads/ 접속 (또는 brew/pacman으로 설치)"
    echo "  2. Python 3.10 이상 설치"
    echo "  3. 설치 후 이 파일을 다시 실행"
    echo
    exit 1
fi

PYVER="$("$PYTHON_BIN" --version 2>&1)"
echo "  파이썬 확인됨: $PYVER"
echo

if [ -f ".venv/bin/activate" ]; then
    echo "  가상환경이 이미 있습니다. 건너뜁니다."
else
    echo "  가상환경을 만드는 중..."
    if ! "$PYTHON_BIN" -m venv .venv; then
        echo
        echo "  [오류] 가상환경 생성에 실패했습니다."
        echo "  폴더 경로에 한글이나 공백이 많으면 문제가 될 수 있습니다."
        echo
        exit 1
    fi
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo
echo "  pip 업그레이드 중..."
python -m pip install --upgrade pip --quiet

echo
echo "------------------------------------------------------------"

OS="$(uname -s)"

if [ "$OS" = "Darwin" ]; then
    echo "  macOS에서는 Apple Silicon(MPS) 가속이 기본으로 지원되므로"
    echo "  NVIDIA GPU 선택 없이 기본 버전을 설치합니다."
    echo
    echo "  PyTorch 설치 중... 시간이 걸릴 수 있습니다."
    if ! pip install torch torchvision; then
        echo
        echo "  [오류] 패키지 설치에 실패했습니다."
        echo "  인터넷 연결을 확인하세요."
        echo
        exit 1
    fi
else
    echo "  NVIDIA 그래픽카드가 있습니까?"
    echo
    echo "     1 = 예       GPU 버전 설치. 학습이 훨씬 빠릅니다."
    echo "     2 = 아니오   CPU 버전 설치. 모르겠으면 2번을 고르세요."
    echo
    read -r -p "  번호 입력 : " GPUCHOICE

    if [ "$GPUCHOICE" = "1" ]; then
        echo
        echo "  GPU 버전 PyTorch 설치 중... 시간이 오래 걸립니다."
        if ! pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121; then
            echo
            echo "  [오류] 패키지 설치에 실패했습니다."
            echo "  인터넷 연결과 방화벽을 확인하세요."
            echo
            exit 1
        fi
    else
        echo
        echo "  CPU 버전 PyTorch 설치 중... 시간이 오래 걸립니다."
        if ! pip install torch torchvision; then
            echo
            echo "  [오류] 패키지 설치에 실패했습니다."
            echo "  인터넷 연결과 방화벽을 확인하세요."
            echo
            exit 1
        fi
    fi
fi

echo
echo "  나머지 패키지 설치 중..."
if ! pip install timm numpy pandas scikit-learn pillow opencv-python matplotlib PyYAML; then
    echo
    echo "  [오류] 패키지 설치에 실패했습니다."
    echo "  인터넷 연결을 확인하세요."
    echo
    exit 1
fi

echo
echo "============================================================"
echo "  설치 확인"
echo "============================================================"
if ! python -c "import torch,timm,cv2,sklearn,yaml;print('  PyTorch  :',torch.__version__);print('  GPU 사용 :',torch.cuda.is_available());print('  timm     :',timm.__version__);print('  OpenCV   :',cv2.__version__)"; then
    echo
    echo "  [오류] 설치는 됐지만 불러오기에 실패했습니다."
    echo
    exit 1
fi

echo
echo "============================================================"
echo "  설치가 끝났습니다."
echo
echo "  다음부터는 ./start.sh 를 실행해서 연구를 시작하세요."
echo "============================================================"
echo
