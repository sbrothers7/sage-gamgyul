"""
core.py — 연구 파이프라인의 핵심 모듈

여기에는 다음이 들어 있습니다.
  1. 설정 읽기 / 재현성(시드) 고정
  2. 데이터셋 스캔 및 학습·검증·시험 분할
  3. 모델 생성 (전이학습)
  4. 성능 지표 계산 (정확도, 정밀도, 재현율, F1, MCC)

실행 스크립트(run_*.py)가 이 모듈을 가져다 씁니다.
직접 실행할 일은 없습니다.
"""

from __future__ import annotations

import json
import os
import platform
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# 프로젝트 최상위 폴더 (이 파일의 부모의 부모)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# =====================================================================
#  1. 설정 및 재현성
# =====================================================================

def load_config(path: str | Path | None = None) -> dict:
    """config.yaml 을 읽어서 딕셔너리로 돌려줍니다."""
    if path is None:
        path = PROJECT_ROOT / "config.yaml"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"설정 파일을 찾을 수 없습니다: {path}\n"
            f"프로젝트 폴더 안에서 실행하고 있는지 확인하세요."
        )
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 상대 경로를 절대 경로로 바꿔 둡니다.
    cfg["data"]["source_A"] = str(PROJECT_ROOT / cfg["data"]["source_A"])
    cfg["data"]["source_B"] = str(PROJECT_ROOT / cfg["data"]["source_B"])
    cfg["output"]["root"] = str(PROJECT_ROOT / cfg["output"]["root"])
    return cfg


def apply_model_override(cfg: dict, model_name: str | None,
                         tag: str | None = None) -> Path:
    """--model 로 architecture를 바꿀 때 씁니다.

    cfg["model"]["name"]을 덮어쓰고, 결과를 모델별 하위 폴더
    (results/<model_name>/)에 저장하도록 경로를 돌려줍니다. 이렇게 해야
    여러 architecture를 비교할 때 서로의 checkpoint·성적표를 덮어쓰지
    않습니다. model_name이 없으면(플래그 생략) 기존과 동일하게 results/
    바로 아래를 씁니다.

    tag 는 같은 architecture로 조건만 바꿔 여러 번 돌릴 때 씁니다
    (예: --tag frac25 → results/<model_name>_frac25/).
    """
    out_root = Path(cfg["output"]["root"])
    if model_name:
        cfg["model"]["name"] = model_name
    name = cfg["model"]["name"] if (model_name or tag) else None
    if name:
        out_root = out_root / (f"{name}_{tag}" if tag else name)
    return out_root


def set_seed(seed: int) -> None:
    """난수를 고정해서 매번 같은 결과가 나오게 합니다.

    재현성은 이 연구의 핵심 주장 중 하나이므로 반드시 호출해야 합니다.
    (Pineau et al., 2021 / Kapoor & Narayanan, 2023 참고)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """GPU가 있으면 GPU를, 없으면 CPU를 씁니다."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_environment() -> dict:
    """실험 환경을 기록합니다. 연구노트에 그대로 붙여 넣으세요."""
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        info["gpu"] = "Apple Silicon (MPS)"
    else:
        info["gpu"] = "없음 (CPU 사용)"
    return info


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


# =====================================================================
#  2. 데이터
# =====================================================================

@dataclass
class ScanResult:
    """폴더를 훑은 결과."""
    frame: pd.DataFrame
    missing_classes: list = field(default_factory=list)
    thin_classes: list = field(default_factory=list)
    total: int = 0


def scan_source(root: str | Path, classes: list[str], source_name: str,
                min_per_class: int = 5) -> ScanResult:
    """data/raw/<source>/<병해이름>/*.jpg 구조를 훑어 표로 만듭니다."""
    root = Path(root)
    rows, missing, thin = [], [], []

    for label_index, cls in enumerate(classes):
        cls_dir = root / cls
        if not cls_dir.is_dir():
            missing.append(cls)
            continue
        files = [p for p in sorted(cls_dir.rglob("*"))
                 if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()]
        if 0 < len(files) < min_per_class:
            thin.append((cls, len(files)))
        if not files:
            missing.append(cls)
        for p in files:
            rows.append({
                "path": str(p),
                "label": cls,
                "label_index": label_index,
                "source": source_name,
            })

    frame = pd.DataFrame(rows, columns=["path", "label", "label_index", "source"])
    return ScanResult(frame=frame, missing_classes=missing,
                      thin_classes=thin, total=len(frame))


def stratified_split(frame: pd.DataFrame, ratios: dict, seed: int) -> pd.DataFrame:
    """병해별 비율을 유지하면서 학습/검증/시험으로 나눕니다.

    '층화(stratified)' 분할이라고 부릅니다. 병해마다 장수가 다를 때
    한쪽에만 몰리는 것을 막아 줍니다.
    """
    total_ratio = ratios["train"] + ratios["val"] + ratios["test"]
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"분할 비율의 합이 1.0이 아닙니다: {total_ratio}")

    rng = np.random.default_rng(seed)
    out = []
    for label, group in frame.groupby("label", sort=True):
        group = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(group)
        n_train = int(round(n * ratios["train"]))
        n_val = int(round(n * ratios["val"]))
        # 남는 것은 전부 test 로
        n_train = min(n_train, max(n - 2, 1)) if n >= 3 else max(n - 2, 1)
        n_val = min(n_val, max(n - n_train - 1, 0))
        splits = (["train"] * n_train + ["val"] * n_val
                  + ["test"] * (n - n_train - n_val))
        group = group.copy()
        group["split"] = splits[:n]
        out.append(group)
        del rng  # 사용하지 않음 (재현성은 random_state로 확보)
        rng = np.random.default_rng(seed)
    return pd.concat(out, ignore_index=True)


def build_transforms(cfg: dict, training: bool):
    """이미지 전처리 파이프라인을 만듭니다."""
    size = cfg["data"]["image_size"]
    # ImageNet 사전학습 모델의 표준 정규화 값
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    if not training:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            normalize,
        ])

    aug = cfg["train"]["augment"]
    steps = [transforms.Resize((size, size))]
    if aug.get("horizontal_flip"):
        steps.append(transforms.RandomHorizontalFlip())
    if aug.get("vertical_flip"):
        steps.append(transforms.RandomVerticalFlip())
    if aug.get("rotation_degrees"):
        steps.append(transforms.RandomRotation(aug["rotation_degrees"]))
    if aug.get("color_jitter"):
        j = aug["color_jitter"]
        steps.append(transforms.ColorJitter(brightness=j, contrast=j, saturation=j))
    steps += [transforms.ToTensor(), normalize]
    return transforms.Compose(steps)


class CitrusDataset(Dataset):
    """표(DataFrame)를 받아서 이미지와 정답을 돌려주는 데이터셋."""

    def __init__(self, frame: pd.DataFrame, transform=None):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label_index"]), str(row["path"])


def make_loader(frame: pd.DataFrame, cfg: dict, training: bool,
                shuffle: bool | None = None) -> DataLoader:
    dataset = CitrusDataset(frame, transform=build_transforms(cfg, training))
    if shuffle is None:
        shuffle = training
    return DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=shuffle,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )


# =====================================================================
#  3. 모델
# =====================================================================

def build_model(cfg: dict, num_classes: int) -> nn.Module:
    """전이학습 모델을 만듭니다.

    ImageNet으로 미리 학습된 가중치를 가져와 마지막 분류층만
    우리 병해 개수에 맞게 새로 답니다. 이것이 '전이학습'입니다.
    """
    import timm

    name = cfg["model"]["name"]
    want_pretrained = cfg["model"]["pretrained"]

    try:
        model = timm.create_model(
            name,
            pretrained=want_pretrained,
            num_classes=num_classes,
            drop_rate=cfg["model"].get("dropout", 0.0),
        )
    except Exception as exc:
        if not want_pretrained:
            raise
        # 사전학습 가중치는 인터넷에서 내려받습니다. 실패하면 원인을 알려줍니다.
        print()
        print("!" * 62)
        print("  사전학습 가중치를 내려받지 못했습니다.")
        print(f"  원인: {type(exc).__name__}: {exc}")
        print()
        print("  확인할 것:")
        print("    1. 인터넷에 연결되어 있는지")
        print("    2. 회사/학교 방화벽이 huggingface.co 를 막고 있지 않은지")
        print()
        print("  해결이 어렵다면 config.yaml 에서 pretrained: false 로 바꿀 수 있지만,")
        print("  전이학습 없이 처음부터 학습하면 데이터가 적을 때 성능이 크게 떨어집니다.")
        print("  원 논문도 전이학습을 사용했으므로 재현을 위해서는 true 가 맞습니다.")
        print("!" * 62)
        raise SystemExit(1) from exc

    return model


def load_checkpoint(path: str | Path, cfg: dict, num_classes: int,
                    device: torch.device) -> nn.Module:
    """저장된 학습 결과를 불러옵니다."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"학습된 모델 파일이 없습니다: {path}\n"
            f"먼저 run_2_train.py 를 실행하세요."
        )
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    # 학습 당시의 architecture를 그대로 씁니다 (config.yaml이 그 사이 바뀌어도 안전).
    ckpt_cfg = {**cfg, "model": {**cfg["model"], "name": checkpoint.get("model_name", cfg["model"]["name"])}}
    model = build_model(ckpt_cfg, num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


# =====================================================================
#  4. 성능 지표
# =====================================================================

def compute_metrics(y_true, y_pred, classes: list[str]) -> dict:
    """정확도·정밀도·재현율·F1·MCC 를 모두 계산합니다.

    MCC(매튜스 상관계수)를 함께 보고하는 이유:
      정확도와 F1은 병해별 장수가 불균형할 때 실제보다 부풀려집니다.
      MCC는 네 칸(TP/FN/TN/FP)이 모두 좋아야 높아집니다.
      → Chicco & Jurman (2020) 근거
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(set(y_true)) > 1 else 0.0,
    }

    # 병해별 세부 성적
    per_class = {}
    labels = list(range(len(classes)))
    p = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    r = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    f = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    for i, cls in enumerate(classes):
        support = int((y_true == i).sum())
        per_class[cls] = {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": support,
        }
    metrics["per_class"] = per_class
    metrics["confusion_matrix"] = confusion_matrix(
        y_true, y_pred, labels=labels
    ).tolist()
    metrics["classes"] = classes
    return metrics


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    """모델로 예측을 수행하고 (정답, 예측, 확률, 파일경로)를 돌려줍니다."""
    model.eval()
    all_true, all_pred, all_prob, all_paths = [], [], [], []
    for images, labels, paths in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        all_true.extend(labels.tolist())
        all_pred.extend(preds.cpu().tolist())
        all_prob.extend(probs.cpu().tolist())
        all_paths.extend(list(paths))
    return all_true, all_pred, all_prob, all_paths


def print_metrics(title: str, metrics: dict) -> None:
    """사람이 읽기 좋게 성적표를 출력합니다."""
    print(f"\n{'=' * 62}")
    print(f"  {title}")
    print(f"{'=' * 62}")
    print(f"  이미지 수      : {metrics['n_samples']}장")
    print(f"  정확도         : {metrics['accuracy'] * 100:6.2f} %")
    print(f"  정밀도(macro)  : {metrics['precision_macro'] * 100:6.2f} %")
    print(f"  재현율(macro)  : {metrics['recall_macro'] * 100:6.2f} %")
    print(f"  F1(macro)      : {metrics['f1_macro'] * 100:6.2f} %")
    print(f"  MCC            : {metrics['mcc']:6.3f}   (1.0이 완벽, 0.0이 찍기 수준)")
    print(f"  {'-' * 58}")
    print(f"  {'병해':<14}{'정밀도':>9}{'재현율':>9}{'F1':>9}{'장수':>7}")
    for cls, m in metrics["per_class"].items():
        print(f"  {cls:<14}{m['precision']:>9.3f}{m['recall']:>9.3f}"
              f"{m['f1']:>9.3f}{m['support']:>7}")
    print(f"{'=' * 62}\n")
