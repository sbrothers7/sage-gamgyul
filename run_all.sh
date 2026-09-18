#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# 선택: ./run_all.sh resnet50 처럼 첫 인자로 timm 모델 이름을 주면
# 그 architecture로 2~5단계를 돌리고 results/<모델이름>/ 에 따로 저장합니다.
MODEL_ARGS=()
if [ "${1:-}" != "" ]; then
    MODEL_ARGS=(--model "$1")
fi

if [ ! -f ".venv/bin/activate" ]; then
    echo "  [오류] 연구 환경이 아직 설치되지 않았습니다. 먼저 ./setup.sh 를 실행하세요."
    exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python run_1_prepare.py
for script in run_2_train.py run_3_evaluate.py run_4_gradcam.py run_5_report.py; do
    echo
    echo "############################################################"
    echo "  실행: $script ${MODEL_ARGS[*]+${MODEL_ARGS[*]}}"
    echo "############################################################"
    python "$script" "${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"}"
done

echo
echo "모든 단계 완료. results/ 를 확인하세요."
