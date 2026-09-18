# 감귤 병해 영상분류 AI의 일반화 성능 분석

> 발표된 감귤 병해 AI 모델을 직접 구현하고, 학습에 쓰이지 않은
> 새 감귤 이미지에서도 성능이 유지되는지 검증하는 연구입니다.

---

## 처음이라면 이 순서대로

### 1. 환경 설치 (딱 한 번만)

`setup_windows.bat` 을 **더블클릭**하세요.

파이썬이 없다면 먼저 https://www.python.org/downloads/ 에서 3.10 이상을
설치하세요. 설치 화면에서 **"Add Python to PATH"** 를 반드시 체크해야 합니다.

설치 중에 "NVIDIA 그래픽카드가 있습니까?"를 묻습니다.
모르겠으면 2번(CPU)을 고르세요. 나중에 다시 설치할 수 있습니다.

### 2. 코드가 도는지 먼저 확인 (데이터 없이)

진짜 감귤 사진을 구하기 전에, 코드가 제대로 도는지 가짜 데이터로 확인합니다.

`start.bat` 을 더블클릭한 뒤, 검은 창에 다음을 순서대로 입력하세요.

```
python make_test_data.py
python run_1_prepare.py
python run_2_train.py
python run_3_evaluate.py
python run_4_gradcam.py
python run_5_report.py
```

전부 오류 없이 끝나면 준비가 된 것입니다.
**여기서 나오는 숫자는 연구 결과가 아닙니다.** 가짜 그림으로 만든
배관 점검용이므로, 확인이 끝나면 반드시 지우세요.

```
python make_test_data.py --clean
```

### 3. 진짜 데이터 넣기

지금까지 쓴 공개 데이터셋은 한 번에 받아서 정리할 수 있습니다 (약 2.3GB).

```
pip install kagglehub
python build_data.py
```

직접 찍은 사진을 더 넣거나 다른 데이터셋을 쓰려면
`docs/01_data_collection_guide.md` 를 읽고 그대로 따라 하세요.
이 연구에서 가장 시간이 오래 걸리는 단계입니다.

### 4. 실험 시작

`docs/02_experiment_protocol.md` 를 읽고 1단계부터 순서대로 실행하세요.

---

## 폴더 설명

```
엄시울ksef/
├── README.md               ← 지금 읽고 있는 문서
├── config.yaml             ← 실험 설정. 여기만 고치면 됩니다
├── setup_windows.bat       ← 환경 설치 (한 번만)
├── start.bat               ← 연구 시작할 때 더블클릭
├── make_test_data.py       ← 가짜 데이터 생성 (코드 점검용)
├── build_data.py           ← 공개 데이터셋 받기 + 중복 정리 → data/raw
│
├── run_1_prepare.py        ← [1] 데이터 준비 및 점검
├── run_2_train.py          ← [2] 모델 학습 (원 논문 재현)
├── run_3_evaluate.py       ← [3] 일반화 성능 검증  ★핵심★
├── run_4_gradcam.py        ← [4] 판단 근거 시각화
├── run_5_report.py         ← [5] 그림과 표 생성
│
├── src/
│   ├── core.py             ← 데이터·모델·성능지표
│   └── explain.py          ← Grad-CAM·배경제거
│
├── docs/
│   ├── 01_data_collection_guide.md
│   ├── 02_experiment_protocol.md
│   └── 03_research_notebook.md  ← 실험할 때마다 기록
│
├── data/raw/
│   ├── source_A/           ← 재현용 (공개 데이터셋)
│   └── source_B/           ← 검증용 (새 감귤 이미지)
│
└── results/                ← 실행하면 자동으로 생깁니다
    ├── checkpoints/        ← 학습된 모델
    ├── metrics/            ← 성적표 (json, csv)
    ├── figures/            ← 논문에 넣을 그림
    ├── gradcam/            ← 히트맵 이미지
    └── 05_결과요약.md       ← 최종 요약
```

---

## 이 연구의 구조

```
        [공개 데이터셋 A]              [새 감귤 이미지 B]
               |                              |
        학습 / 검증 / 시험                 평가에만 사용
               |                              |
               v                              |
         모델 학습 (전이학습)                  |
               |                              |
               +--------------+---------------+
                              |
                              v
                     같은 모델, 두 가지 데이터
                              |
              +---------------+---------------+
              v                               v
       재현 성능 (A 시험셋)            일반화 성능 (B 전체)
              |                               |
              +---------------+---------------+
                              v
                    ★ 일반화 격차 = 차이 ★
```

`source_B`는 **절대 학습에 쓰지 않습니다.** 이 원칙이 깨지면 연구가 무효입니다.

---

## 나오는 숫자들의 뜻

| 지표 | 뜻 | 주의 |
|---|---|---|
| 정확도 | 전체 중 맞힌 비율 | 병해별 장수가 다르면 부풀려집니다 |
| 정밀도 | "병이라 했을 때" 실제로 병인 비율 | 오진(허위 경보)을 봅니다 |
| 재현율 | 실제 병 중 찾아낸 비율 | 놓친 병을 봅니다 |
| F1 | 정밀도와 재현율의 조화평균 | 둘의 균형 |
| **MCC** | 네 칸이 모두 좋아야 높아지는 지표 | **불균형에 속지 않음** |

MCC를 함께 보는 이유는 Chicco & Jurman (2020)의 근거를 따른 것입니다.
정확도와 F1은 병해별 장수가 불균형할 때 실제보다 좋아 보입니다.

| 잎 집중도 | 뜻 |
|---|---|
| 1에 가까움 | 모델이 **잎**을 보고 판단 (정상) |
| 0에 가까움 | 모델이 **배경**을 보고 판단 (문제) |

---

## 자주 나는 오류

**`ModuleNotFoundError: No module named 'torch'`**
환경이 활성화되지 않았습니다. `start.bat` 으로 실행하세요.

**`CUDA out of memory`**
`config.yaml` 에서 `batch_size` 를 8이나 4로 줄이세요.

**학습이 너무 느림**
GPU가 없으면 오래 걸립니다. `config.yaml` 에서 `epochs` 를 10 정도로 줄이거나,
`model.name` 을 `mobilenetv3_large_100` 으로 바꾸세요. 훨씬 가볍습니다.

**사전학습 가중치 다운로드 실패**
인터넷 연결이나 학교 방화벽 문제입니다. 화면에 안내가 나옵니다.

**폴더를 찾을 수 없다는 오류**
`config.yaml` 의 `classes` 와 실제 폴더 이름이 다릅니다.
대소문자까지 정확히 같아야 합니다.

**`.bat` 실행 시 한글이 깨지고 이상한 명령어 오류가 남**

예: `'씠'은(는) 내부 또는 외부 명령... 이 아닙니다`

배치 파일은 한국어 Windows에 맞춰 CP949로 저장되어 있습니다.
그래도 깨진다면 Windows 설정에서 "세계 언어 지원을 위해 Unicode UTF-8 사용"이
켜져 있는 것입니다. 이 경우 아래 **수동 설치**를 쓰세요.

---

## 수동 설치 (배치 파일이 안 될 때)

`setup_windows.bat` 없이 직접 설치하는 방법입니다.
시작 메뉴에서 **명령 프롬프트**를 열고 아래를 한 줄씩 입력하세요.

**1. 프로젝트 폴더로 이동**

```
D:
cd "D:\TestRuns\_AI르네상스-학원\엄시울ksef"
```

**2. 가상환경 만들기 (한 번만)**

```
python -m venv .venv
```

**3. 가상환경 켜기 (매번 필요)**

```
.venv\Scripts\activate
```

성공하면 줄 앞에 `(.venv)` 가 붙습니다.

**4. 패키지 설치 (한 번만)**

GPU가 없다면:

```
pip install torch torchvision
```

NVIDIA 그래픽카드가 있다면:

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

이어서 공통으로:

```
pip install timm numpy pandas scikit-learn pillow opencv-python matplotlib PyYAML
```

**5. 설치 확인**

```
python -c "import torch, timm, cv2; print(torch.__version__, torch.cuda.is_available())"
```

버전 번호와 `True`/`False` 가 나오면 성공입니다.

**6. 다음부터 연구할 때는 3번만 다시 하면 됩니다.**

```
D:
cd "D:\TestRuns\_AI르네상스-학원\엄시울ksef"
.venv\Scripts\activate
python run_1_prepare.py
```

> 파이썬 스크립트(`run_*.py`)의 한글 출력은 인코딩 문제가 없습니다.
> 파이썬은 Windows 콘솔에 유니코드로 직접 쓰기 때문입니다.
> 문제가 되는 것은 `.bat` 파일뿐입니다.

---

## 근거 논문

이 코드의 설계는 다음 논문들을 따랐습니다.

- **재현 대상** — Devora-Guadarrama, M., et al. (2025). Deep learning-based citrus canker and Huanglongbing disease detection using leaf images. *Computers, 14*(11), 500.
- **실험 설계** — Mahapatra, P., et al. (2026). Advancing plant disease classification using an attention-based CNN for intra-dataset and cross-dataset training. *Scientific Reports, 16*, 10925.
- **일반화 격차** — Ahmad, A., El Gamal, A., & Saraswat, D. (2023). Toward generalization of deep learning-based plant disease identification. *IEEE Access, 11*, 9042–9057.
- **판단 근거 시각화** — Selvaraju, R. R., et al. (2017). Grad-CAM. *ICCV 2017*, 618–626.
- **성능 지표 선택** — Chicco, D., & Jurman, G. (2020). The advantages of MCC over F1 score and accuracy. *BMC Genomics, 21*, 6.
- **재현성·데이터 누출** — Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis. *Patterns, 4*(9), 100804.
