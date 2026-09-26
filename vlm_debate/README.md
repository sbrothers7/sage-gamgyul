# 감귤 병해 진단 — 시각-언어 모델(VLM) 다중 에이전트 토론 실험

VIDA(독립 진단) → PANDA(이름을 부르는 동료 토론) 파이프라인
(Al Juboori et al., 2026, *Frontiers in Plant Science*)을 감귤 잎 데이터에 맞춰
다시 구현하고, **반-아첨(anti-sycophancy) 장치**의 효과를 제거 실험으로 검증한 코드입니다.

- 같은 리포지토리 루트의 CNN 파이프라인(run_1~run_5)은 "공개 데이터로 학습한 CNN이 다른
  출처 사진에서 무너지는가"를 보는 기준 실험이고, 이 폴더는 학습 없이 진단하는 VLM 쪽 실험입니다.
- 모든 모델은 **로컬(Ollama)** 에서 실행합니다. 상용 API 모델은 예고 없이 폐기되어
  나중에 재현이 불가능하기 때문입니다.

## macOS · Linux (Arch 포함)

```bash
cd vlm_debate
bash run.sh selftest      # 1) 코드 점검 — 모델·데이터 없이 몇 초
```

**1. Ollama 설치 (한 번만)**

| OS | 명령 |
|---|---|
| macOS | `brew install ollama` (또는 https://ollama.com/download 앱) |
| Arch (NVIDIA) | `sudo pacman -S ollama-cuda` |
| Arch (AMD / CPU) | `sudo pacman -S ollama-rocm` / `sudo pacman -S ollama` |
| 기타 Linux | `curl -fsSL https://ollama.com/install.sh \| sh` |

서버가 꺼져 있으면 `run.sh` 가 `ollama serve` 를 백그라운드로 띄웁니다
(Arch 에서 서비스로 쓰려면 `sudo systemctl enable --now ollama`).

**2. 준비와 실험**

```bash
bash run.sh prepare       # 모델 3종 + Mendeley·Zenodo 다운로드(약 15 GB), 중복 검사
bash run.sh finalize      # Zenodo 768px 변환, 모델 선정 확인, 10장 파일럿
bash run.sh main          # 데이터셋 A: ON → OFF   (GPU 기준 7~8시간)
bash run.sh night2        # 데이터셋 A: HARD, 데이터셋 B: ON
bash run.sh analyze       # results/ANALYSIS.md
bash run.sh app           # 웹 콘솔 http://localhost:8765
```

- 파이썬 패키지는 처음 실행할 때 `.venv/` 에 자동 설치됩니다 (Arch·Homebrew 의 시스템 pip 제한 회피).
- 이미지는 `~/citrus-data` 에 저장됩니다. 다른 곳을 쓰려면 `export CITRUS_DATA=/경로` 또는 `.env` 에 적습니다.
  `data/*.json` 의 경로는 이 폴더 기준 **상대경로**라서 OS 에 상관없이 그대로 씁니다.
- 같은 seed 면 Windows·macOS·Linux 에서 **같은 이미지가 뽑히도록** 정렬 순서를 맞춰 두었습니다.
- Apple Silicon 은 GPU(Metal)를 자동으로 씁니다. 모델 속도는 `bash run.sh bench` 로 확인하세요.

**3. 제주 온주밀감 데이터로 실험하기**

1. 사진을 병해별 폴더로 정리: `~/citrus-data/jeju/Healthy/*.jpg`, `~/citrus-data/jeju/Canker/*.jpg` …
2. 명세 파일 `data/jeju.json` 작성:
   ```json
   { "crop": "Citrus",
     "classes": { "Healthy": ["jeju/Healthy"], "Canker": ["jeju/Canker"] } }
   ```
3. 세 조건을 같은 R1 답으로 실행 (첫 실행의 R1 을 뒤 두 실행이 재사용):
   ```bash
   bash run.sh run --dataset data/jeju.json --n 100 --tag jeju_ev-on
   R1=$(ls -d results/*_jeju_ev-on | tail -1)/vida_raw_bank.json
   bash run.sh run --dataset data/jeju.json --n 100 --no-evidence --reuse-r1 "$R1" --tag jeju_ev-off
   bash run.sh run --dataset data/jeju.json --n 100 --enforce     --reuse-r1 "$R1" --tag jeju_ev-hard
   python3 tools/analyze.py results jeju      # 제주 실행만 모아 분석
   ```
   `--n` 은 상한입니다. 층화 추출(병해 80 / 건강 20) 때문에 실제 장수는 클래스별 사진 수에 따라 줄 수 있습니다.

## 실행 순서 (Windows, 더블클릭)

| 파일 | 하는 일 |
|---|---|
| `0_selftest.bat` | 가짜 에이전트로 코드 점검 (키·데이터 불필요) |
| `4_prepare_all.bat` | Ollama 설치, 모델 다운로드, Mendeley·Zenodo 데이터 다운로드, 중복 검사 |
| `5_diagnose.bat` | 모델별 속도·형식 준수·GPU 사용 여부 진단 |
| `7_gpu_fix.bat` | Ollama 가 GPU 를 못 잡을 때 복구 (VC++ 런타임 재설치 등) + 모델 선정 + 10장 파일럿 |
| `8_main_overnight.bat` | 본 실험 1: 데이터셋 A 190장, 반-아첨 ON → OFF |
| `9_night2.bat` | 본 실험 2: 데이터셋 A HARD(코드 강제) + 데이터셋 B(Zenodo) ON |
| `3_run.bat` | 웹 콘솔 (http://localhost:8765) |

분석: `py tools\analyze.py` → `results\ANALYSIS.md`
직접 실행: `py tools\run_experiment.py --dataset data\jeju.json --n 100 --tag jeju_ev-on` (옵션은 아래 macOS·Linux 절과 같음)

## 실험 조건

| 조건 | 내용 |
|---|---|
| OFF | 답 변경 제한 없음 |
| ON | R2 지시문에 "새 시각 근거가 있을 때만 바꿀 것" 명시 (원 논문 방식) |
| HARD | ON + 새 근거 없이 바꾼 답을 **코드가 되돌림** |

세 조건은 같은 이미지와 **같은 R1 답**(`--reuse-r1`)을 공유하므로, 차이는 토론 규칙에서만 생깁니다.

## 주요 결과 (공개 데이터)

| 데이터 | 조건 | 토론 전 → 후 | Δ | McNemar p |
|---|---|---|---|---|
| A (Mendeley, 4클래스, 190장) | OFF | 0.368 → 0.384 | +0.016 | 0.45 |
| A | ON | 0.368 → 0.389 | +0.021 | 0.34 |
| A | HARD | 0.368 → 0.368 | 0.000 | 1.00 |
| B (CitrusUAT, 12클래스, 180장) | ON | 0.189 → 0.206 | +0.017 | 0.55 |

- 지시문 규칙은 답 변경의 66–86% 에서 무시됨 (새 근거 "NONE" 인 채로 변경)
- 약한 모델(llava:7b)의 향상(0.284 → 0.43)은 HARD 에서 사라짐(→ 0.305) → 향상의 출처는 **근거 없는 동조**
- 세부: `results/ANALYSIS.md`

## 폴더

```
src/        vp_core.py (파이프라인 본체), vp_app.py (웹 콘솔), prepare_dataset.py
tools/      prepare_all, diagnose, gpu_check, run_experiment, run_main, run_night2, analyze ...
data/       데이터셋 명세 json (이미지는 C:\citrus-data 에 있고 경로만 참조)
results/    실험별 결과 (CSV, 토론 원문, summary.json, ANALYSIS.md)
```

## 환경

RTX 3070 Laptop (8 GB), Ryzen 7 5800H, Ollama 0.34.3,
모델: llava:7b, qwen2.5vl:7b, qwen2.5vl:3b (temperature 0.2, max 700 tokens), seed 42.
