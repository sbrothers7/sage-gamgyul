"""
[5단계] 결과 정리 — 그림과 표 만들기

하는 일:
  - 학습 곡선 그래프
  - 혼동행렬 (어떤 병해를 어떤 병해로 착각했는지)
  - 조건별 성능 비교 막대그래프  <- 논문의 대표 그림
  - 결과 요약 마크다운 문서

  여기서 나온 그림을 그대로 보고서에 넣으면 됩니다.

실행:
  python run_5_report.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.core import apply_model_override, ensure_dir, load_config

# 그림 공통 스타일
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

CONDITION_LABELS = {
    "A_test": "Same source\n(reproduction)",
    "B": "New citrus images\n(generalization)",
    "A_test_nobg": "Same source\n+ bg removed",
    "B_nobg": "New images\n+ bg removed",
}
COLORS = {
    "A_test": "#4C78A8",
    "B": "#E45756",
    "A_test_nobg": "#9ECAE9",
    "B_nobg": "#FF9D98",
}


def plot_training_curve(out_root: Path, fig_dir: Path) -> bool:
    path = out_root / "02_train_history.csv"
    if not path.exists():
        return False
    history = pd.read_csv(path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["epoch"], history["train_loss"], color="#4C78A8", marker="o",
                 markersize=3, label="train loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training loss")
    axes[0].legend(frameon=False)

    axes[1].plot(history["epoch"], history["train_acc"] * 100, color="#4C78A8",
                 marker="o", markersize=3, label="train accuracy")
    if history["val_acc"].notna().any():
        axes[1].plot(history["epoch"], history["val_acc"] * 100, color="#E45756",
                     marker="s", markersize=3, label="validation accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0, 103)
    axes[1].set_title("Accuracy")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_training_curve.png")
    plt.close(fig)
    return True


def plot_confusion(metrics: dict, classes: list, title: str, path: Path) -> None:
    matrix = np.array(metrics["confusion_matrix"], dtype=float)
    if matrix.sum() == 0:
        return
    # 행 기준 백분율로 정규화
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix),
                           where=row_sums > 0) * 100

    fig, ax = plt.subplots(figsize=(1.6 * len(classes) + 2.4,
                                    1.4 * len(classes) + 2.0))
    im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=11)
    ax.grid(False)

    for i in range(len(classes)):
        for j in range(len(classes)):
            value = normalized[i, j]
            ax.text(j, i, f"{int(matrix[i, j])}\n{value:.0f}%",
                    ha="center", va="center", fontsize=9,
                    color="white" if value > 55 else "#333333")

    fig.colorbar(im, ax=ax, shrink=0.8, label="% of actual class")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_comparison(results: dict, fig_dir: Path) -> bool:
    keys = [k for k in CONDITION_LABELS if k in results]
    if len(keys) < 2:
        return False

    metrics_to_plot = [("accuracy", "Accuracy (%)", 100),
                       ("f1_macro", "Macro F1", 1),
                       ("mcc", "MCC", 1)]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))

    for ax, (key, label, scale) in zip(axes, metrics_to_plot):
        values = [results[k][key] * scale for k in keys]
        bars = ax.bar([CONDITION_LABELS[k] for k in keys], values,
                      color=[COLORS[k] for k in keys], width=0.6)
        ax.set_ylabel(label)
        ax.set_ylim(0, max(scale * 1.08, max(values) * 1.2) if values else scale)
        ax.tick_params(axis="x", labelsize=7.5, pad=2)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value,
                    f"{value:.2f}" if scale == 1 else f"{value:.1f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle("Generalization gap of the citrus disease classifier",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_condition_comparison.png")
    plt.close(fig)
    return True


def plot_per_class_drop(results: dict, classes: list, fig_dir: Path) -> bool:
    if "A_test" not in results or "B" not in results:
        return False
    a = results["A_test"]["per_class"]
    b = results["B"]["per_class"]
    labels, drops = [], []
    for cls in classes:
        if cls in a and cls in b and a[cls]["support"] and b[cls]["support"]:
            labels.append(cls)
            drops.append((a[cls]["f1"] - b[cls]["f1"]))
    if not labels:
        return False

    fig, ax = plt.subplots(figsize=(1.5 * len(labels) + 3, 4))
    colors = ["#E45756" if d > 0 else "#54A24B" for d in drops]
    bars = ax.bar(labels, drops, color=colors, width=0.55)
    ax.axhline(0, color="#444444", linewidth=1)
    ax.set_ylabel("F1 drop  (same source  →  new images)")
    ax.set_title("Which diseases break first on unseen citrus images?",
                 fontsize=11)
    for bar, d in zip(bars, drops):
        ax.text(bar.get_x() + bar.get_width() / 2, d,
                f"{d:+.3f}", ha="center",
                va="bottom" if d >= 0 else "top", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_per_class_drop.png")
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default=None,
        help="run_2_train.py 때와 같은 --model 값을 주면 그 architecture의 "
             "결과(results/<모델이름>/)로 보고서를 만듭니다. 생략하면 config.yaml 값을 씁니다.",
    )
    parser.add_argument(
        "--tag", default=None,
        help="같은 모델로 조건만 바꿔 여러 번 돌릴 때 결과를 results/<모델이름>_<tag>/ 에 따로 저장합니다.",
    )
    args = parser.parse_args()

    cfg = load_config()
    classes = cfg["classes"]
    out_root = apply_model_override(cfg, args.model, args.tag)
    fig_dir = ensure_dir(out_root / "figures")

    print("=" * 62)
    print("  [5단계] 결과 정리")
    print("=" * 62)

    made = []

    if plot_training_curve(out_root, fig_dir):
        made.append("fig1_training_curve.png — 학습 곡선")

    metrics_path = out_root / "metrics" / "03_all_metrics.json"
    results = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            results = json.load(f)

    if "A_test" in results:
        plot_confusion(results["A_test"], classes,
                       "Same source (reproduction)",
                       fig_dir / "fig2a_confusion_same_source.png")
        made.append("fig2a_confusion_same_source.png — 혼동행렬 (재현)")
    if "B" in results:
        plot_confusion(results["B"], classes,
                       "New citrus images (generalization)",
                       fig_dir / "fig2b_confusion_new_images.png")
        made.append("fig2b_confusion_new_images.png — 혼동행렬 (일반화)")

    if plot_comparison(results, fig_dir):
        made.append("fig3_condition_comparison.png — 조건별 성능 비교 (대표 그림)")
    if plot_per_class_drop(results, classes, fig_dir):
        made.append("fig4_per_class_drop.png — 병해별 성능 하락")

    # ---- 요약 문서 ------------------------------------------------------
    lines = [
        "# 실험 결과 요약",
        "",
        f"- 모델: `{cfg['model']['name']}`",
        f"- 난수 시드: {cfg['seed']}",
        f"- 분류 대상: {', '.join(classes)}",
        "",
        "## 조건별 성능",
        "",
        "| 조건 | 정확도 | 정밀도 | 재현율 | F1 | MCC | 장수 |",
        "|---|---|---|---|---|---|---|",
    ]
    korean = {
        "A_test": "재현 (같은 출처)",
        "B": "일반화 (새 감귤 이미지)",
        "A_test_nobg": "재현 + 배경 제거",
        "B_nobg": "일반화 + 배경 제거",
    }
    for key, name in korean.items():
        if key not in results:
            continue
        m = results[key]
        lines.append(
            f"| {name} | {m['accuracy']*100:.2f}% | {m['precision_macro']*100:.2f}% "
            f"| {m['recall_macro']*100:.2f}% | {m['f1_macro']:.3f} "
            f"| {m['mcc']:.3f} | {m['n_samples']} |"
        )

    if "generalization_gap" in results:
        gap = results["generalization_gap"]
        lines += [
            "",
            "## 핵심 결과",
            "",
            f"**일반화 격차: 정확도 {gap['accuracy_drop_percentage_points']:+.2f}%p, "
            f"F1 {gap['f1_drop']:+.3f}**",
            "",
            "학습에 사용한 데이터와 같은 출처에서 측정한 성능과, 학습에 사용하지 않은",
            "새 감귤 이미지에서 측정한 성능의 차이입니다. 이 값이 이 연구의 답입니다.",
        ]

    cam_summary = out_root / "04_gradcam_summary.json"
    if cam_summary.exists():
        with open(cam_summary, "r", encoding="utf-8") as f:
            cam = json.load(f)
        lines += ["", "## 잎 집중도 (Grad-CAM)", "",
                  "| 출처 | 평균 집중도 | 정답률 | 유효 표본 |", "|---|---|---|---|"]
        any_invalid = False
        for source, m in cam.items():
            focus = m.get("mean_leaf_focus")
            focus_text = f"{focus:.3f}" if focus is not None else "측정 불가"
            n_valid = m.get("n_valid_mask", m.get("n", 0))
            if focus is None or n_valid < m.get("n", 0):
                any_invalid = True
            lines.append(f"| {source} | {focus_text} | "
                         f"{m['accuracy']*100:.1f}% | {n_valid} / {m.get('n', 0)} |")
        lines += ["", "집중도가 1에 가까우면 모델이 잎을 보고 판단한 것이고,",
                  "0에 가까우면 배경을 보고 판단한 것입니다."]
        if any_invalid:
            lines += [
                "",
                "> **주의**: 잎 분리에 실패한 이미지는 평균에서 제외했습니다.",
                "> 배경이 잎과 색이 비슷하면 분리가 실패합니다.",
                "> 유효 표본이 전체의 절반에 못 미치면 이 수치는 보고하지 마세요.",
            ]

    if made:
        lines += ["", "## 생성된 그림", ""]
        lines += [f"- `figures/{m}`" for m in made]

    summary_path = out_root / "05_결과요약.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    for m in made:
        print(f"  생성: {m}")
    print(f"\n  요약 문서: {summary_path}")
    print(f"  그림 폴더: {fig_dir}")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
