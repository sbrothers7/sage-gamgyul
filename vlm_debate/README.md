# 감귤 병해 진단 — 시각-언어 모델(VLM) 다중 에이전트 토론 실험

VIDA(독립 진단) → PANDA(이름을 부르는 동료 토론) 파이프라인
(Al Juboori et al., 2026, *Frontiers in Plant Science*)을 감귤 잎 데이터에 맞춰
다시 구현하고, **반-아첨(anti-sycophancy) 장치**의 효과를 제거 실험으로 검증한 코드입니다.

- 같은 리포지토리 루트의 CNN 파이프라인(run_1~run_5)은 "공개 데이터로 학습한 CNN이 다른
  출처 사진에서 무너지는가"를 보는 기준 실험이고, 이 폴더는 학습 없이 진단하는 VLM 쪽 실험입니다.
- 모든 모델은 **로컬(Ollama)** 에서 실행합니다. 상용 API 모델은 예고 없이 폐기되어
  나중에 재현이 불가능하기 때문입니다.

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
