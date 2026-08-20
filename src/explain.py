"""
explain.py — AI가 무엇을 보고 판단했는지 확인하는 모듈

두 가지 기능이 들어 있습니다.

  1. Grad-CAM
     모델이 이미지의 어느 부분을 근거로 판단했는지 히트맵으로 보여줍니다.
     빨간 부분 = 판단에 크게 기여한 곳.
     병반이 아니라 배경이 빨갛다면, 그 모델은 병을 본 게 아닙니다.
     → Selvaraju et al. (2017)

  2. 배경 제거
     잎만 남기고 배경을 지웁니다. 배경을 지운 뒤 성능이 크게 떨어진다면
     모델이 병반이 아니라 촬영 환경을 학습했다는 뜻입니다.
     → Mahapatra et al. (2026) 의 대조 실험 설계
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
#  1. Grad-CAM
# =====================================================================

def find_last_conv_layer(model: nn.Module) -> nn.Module:
    """모델의 마지막 합성곱 층을 자동으로 찾습니다.

    Grad-CAM은 마지막 합성곱 층에 걸어야 의미 있는 히트맵이 나옵니다.
    (Toda & Okura, 2019 의 층 선택 지침)
    """
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise RuntimeError("합성곱 층을 찾지 못했습니다. 이 모델은 Grad-CAM을 쓸 수 없습니다.")
    return last_conv


class GradCAM:
    """Grad-CAM 히트맵 생성기.

    사용법:
        cam = GradCAM(model)
        heatmap = cam(input_tensor, class_index)
        cam.close()
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer or find_last_conv_layer(model)
        self.activations = None
        self.gradients = None
        self._handles = [
            self.target_layer.register_forward_hook(self._save_activation),
            self.target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor: torch.Tensor,
                 class_index: int | None = None) -> np.ndarray:
        """히트맵을 0~1 범위의 2차원 배열로 돌려줍니다."""
        input_tensor = input_tensor.clone().requires_grad_(True)
        logits = self.model(input_tensor)

        if class_index is None:
            class_index = int(logits.argmax(dim=1).item())

        self.model.zero_grad(set_to_none=True)
        score = logits[0, class_index]
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM 훅이 값을 받지 못했습니다.")

        # 기울기를 전역 평균 풀링 → 채널별 중요도 가중치
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        # 특징맵의 가중합 → ReLU (양의 근거만 남김)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False
        )
        cam = cam[0, 0].cpu().numpy()

        # 0~1로 정규화
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
        return cam

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """정규화된 텐서를 사람이 볼 수 있는 이미지로 되돌립니다."""
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    array = tensor.detach().cpu().numpy()
    if array.ndim == 4:
        array = array[0]
    array = array * std + mean
    array = np.clip(array, 0, 1)
    return np.transpose(array, (1, 2, 0))


def overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.5) -> np.ndarray:
    """원본 이미지 위에 히트맵을 겹칩니다. 결과는 0~255 RGB."""
    image_uint8 = (np.clip(image_rgb, 0, 1) * 255).astype(np.uint8)
    heat_uint8 = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    blended = (1 - alpha) * image_uint8 + alpha * colored
    return np.clip(blended, 0, 255).astype(np.uint8)


def lesion_focus_score(heatmap: np.ndarray, leaf_mask: np.ndarray) -> float:
    """히트맵의 관심이 잎 안쪽에 얼마나 몰려 있는지 0~1로 계산합니다.

    1에 가까울수록 모델이 잎(=병반이 있을 곳)을 봤다는 뜻이고,
    0에 가까울수록 배경을 봤다는 뜻입니다.

    이 수치를 source_A와 source_B에서 각각 구해 비교하면,
    "새 이미지에서 성능이 떨어진 이유"를 숫자로 설명할 수 있습니다.
    """
    if leaf_mask.shape != heatmap.shape:
        leaf_mask = cv2.resize(
            leaf_mask.astype(np.uint8), (heatmap.shape[1], heatmap.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    total = float(heatmap.sum())
    if total <= 0:
        return 0.0
    return float(heatmap[leaf_mask.astype(bool)].sum() / total)


# =====================================================================
#  2. 배경 제거
# =====================================================================

def leaf_mask_with_status(image_bgr: np.ndarray) -> tuple[np.ndarray, bool]:
    """잎 마스크와 '믿을 만한가' 여부를 함께 돌려줍니다.

    두 번째 값이 False면 잎 분리에 실패한 것입니다.
    이때 마스크는 이미지 전체가 되므로, 이 값으로 집중도를 계산하면
    항상 1.0이 나와 버립니다. 반드시 걸러내야 합니다.

    방법: HSV 색공간에서 초록~노랑~갈색 범위를 잡고,
          잡티를 제거한 뒤 가장 큰 덩어리 하나만 남깁니다.
    감귤 잎처럼 배경이 단순한 사진에서 잘 동작합니다.
    배경이 흙·마른 잎처럼 잎과 색이 비슷하면 실패할 수 있습니다.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # 초록 계열 (건전한 잎)
    green = cv2.inRange(hsv, (25, 30, 30), (95, 255, 255))
    # 노랑~갈색 계열 (황화, 괴사, 병반)
    yellow_brown = cv2.inRange(hsv, (5, 40, 40), (35, 255, 255))
    mask = cv2.bitwise_or(green, yellow_brown)

    # 잡티 제거
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # 가장 큰 덩어리만 남기기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cleaned = np.zeros_like(mask)
        cv2.drawContours(cleaned, [largest], -1, 255, thickness=cv2.FILLED)
        mask = cleaned

    # 마스크가 너무 작거나 너무 크면 분리 실패로 판정합니다.
    # (전체를 잎으로 돌려주되, 두 번째 값을 False로 표시)
    ratio = mask.mean() / 255.0
    if ratio < 0.03 or ratio > 0.95:
        return np.ones(mask.shape, dtype=bool), False

    return mask.astype(bool), True


def leaf_mask(image_bgr: np.ndarray) -> np.ndarray:
    """잎 마스크만 필요할 때 쓰는 간편 함수 (배경 제거용)."""
    mask, _ = leaf_mask_with_status(image_bgr)
    return mask


def remove_background(image_bgr: np.ndarray,
                      fill: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """배경을 단색으로 칠한 이미지를 돌려줍니다."""
    mask = leaf_mask(image_bgr)
    output = np.full_like(image_bgr, fill, dtype=np.uint8)
    output[mask] = image_bgr[mask]
    return output


def process_folder(src_root: str | Path, dst_root: str | Path,
                   extensions: set[str] | None = None) -> int:
    """폴더 전체의 배경을 제거해 다른 폴더에 저장합니다.

    폴더 구조(병해별 하위 폴더)는 그대로 유지됩니다.
    """
    if extensions is None:
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    src_root, dst_root = Path(src_root), Path(dst_root)
    count = 0
    for src in sorted(src_root.rglob("*")):
        if not src.is_file() or src.suffix.lower() not in extensions:
            continue
        image = cv2.imread(str(src))
        if image is None:
            continue
        dst = dst_root / src.relative_to(src_root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst.with_suffix(".png")), remove_background(image))
        count += 1
    return count
