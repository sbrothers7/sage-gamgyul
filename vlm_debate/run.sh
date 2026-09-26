#!/usr/bin/env bash
# macOS / Linux 실행 스크립트 (Windows 의 *.bat 과 같은 일을 합니다)
#   ./run.sh selftest   가짜 에이전트로 코드 점검
#   ./run.sh prepare    Ollama 확인, 모델 3종·데이터셋 다운로드, 중복 검사, 파일럿
#   ./run.sh diagnose   모델별 속도·형식·GPU 진단
#   ./run.sh finalize   Zenodo 정리 + 모델 선정 + 10장 파일럿
#   ./run.sh main       본 실험 1: 데이터셋 A, ON → OFF (R1 공유)
#   ./run.sh night2     본 실험 2: 데이터셋 A HARD + 데이터셋 B ON
#   ./run.sh run ...    임의 실행   예) ./run.sh run --dataset data/jeju.json --n 100 --enforce --tag jeju_hard
#   ./run.sh analyze    results/ANALYSIS.md 생성
#   ./run.sh app        웹 콘솔 (http://localhost:8765)
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
mkdir -p logs

# Python: .venv 가 있으면 그것을, 없고 시스템 파이썬에 패키지가 없으면 .venv 를 만든다
# (Arch·Homebrew 파이썬은 시스템 pip 설치를 막기 때문, PEP 668)
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY="${PYTHON:-python3}"; fi
ensure_deps() {
  "$PY" -c "import PIL, openai" 2>/dev/null && return
  if [ ! -x .venv/bin/python ]; then echo "creating .venv ..."; "${PYTHON:-python3}" -m venv .venv; fi
  PY=.venv/bin/python
  "$PY" -m pip install -q --upgrade pip && "$PY" -m pip install -q -r requirements.txt
}
ensure_ollama() {
  if ! curl -s -m 3 http://localhost:11434/api/tags >/dev/null; then
    if command -v ollama >/dev/null; then
      echo "starting 'ollama serve' in the background (log: logs/ollama_serve.log)"
      nohup ollama serve > logs/ollama_serve.log 2>&1 &
      for _ in $(seq 1 30); do curl -s -m 2 http://localhost:11434/api/tags >/dev/null && break; sleep 1; done
    else
      echo "Ollama is not installed. macOS: brew install ollama | Arch: sudo pacman -S ollama-cuda | https://ollama.com"; exit 1
    fi
  fi
}

cmd="${1:-help}"; shift || true
case "$cmd" in
  selftest) ensure_deps; "$PY" tools/selftest/make_fake_data.py && "$PY" tools/selftest/run_selftest.py ;;
  prepare)  ensure_deps; "$PY" -u tools/prepare_all.py ;;
  diagnose) ensure_deps; ensure_ollama; "$PY" -u tools/diagnose.py ;;
  finalize) ensure_deps; ensure_ollama; "$PY" -u tools/finalize.py ;;
  main)     ensure_deps; ensure_ollama; "$PY" -u tools/run_main.py --dataset data/citrusA_mendeley_4cls.json --n 200 ;;
  night2)   ensure_deps; ensure_ollama; "$PY" -u tools/run_night2.py ;;
  run)      ensure_deps; ensure_ollama; "$PY" -u tools/run_experiment.py "$@" ;;
  bench)    ensure_deps; ensure_ollama; "$PY" -u tools/bench_models.py "$@" ;;
  analyze)  "$PY" tools/analyze.py ;;
  app)      ensure_deps; ensure_ollama; "$PY" src/vp_app.py ;;
  *) sed -n '2,12p' "$0" ;;
esac
