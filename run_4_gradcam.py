"""
[4단계] Grad-CAM — 모델이 무엇을 보고 판단했는지 확인

하는 일:
  - 각 병해마다 몇 장씩 골라 히트맵을 만듭니다.
  - 원본 / 히트맵 / 겹친 그림을 나란히 저장합니다.
  - '잎 집중도(lesion focus score)'를 계산합니다.
      1에 가까울수록 모델이 잎을 봤다는 뜻,
      0에 가까울수록 배경을 봤다는 뜻입니다.
  - source_A 와 source_B 의 잎 집중도를 비교합니다.

  성능이 떨어졌다면 왜 떨어졌는지를 이 숫자로 설명할 수 있습니다.

실행:
  python run_4_gradcam.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.core import (
    apply_model_override, build_transforms, ensure_dir, get_device,
    load_checkpoint, load_config, save_json, set_seed,
)
from src.explain import (
    GradCAM, denormalize, leaf_mask_with_status, lesion_focus_score,
    overlay_heatmap,
)
from PIL import Image


def run_source(model, frame, cfg, device, classes, source_key, out_dir):
    """한 출처에 대해 히트맵을 만들고 잎 집중도를 계산합니다."""
    transform = build_transforms(cfg, training=False)
    n_per_class = cfg["gradcam"]["num_samples_per_class"]
    alpha = cfg["gradcam"]["overlay_alpha"]
    records = []

    cam = GradCAM(model)
    try:
        for cls in classes:
            subset = frame[frame["label"] == cls]
            if len(subset) == 0:
                continue
            subset = subset.head(n_per_class)
            for i, (_, row) in enumerate(subset.iterrows()):
                path = Path(row["path"])
                try:
                    pil = Image.open(path).convert("RGB")
                except Exception:
                    continue
                tensor = transform(pil).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(tensor)
                    probs = torch.softmax(logits, dim=1)[0]
                pred_index = int(probs.argmax().item())
                confidence = float(probs[pred_index].item())

                heatmap = cam(tensor, pred_index)
                base = denormalize(tensor)
                blended = overlay_heatmap(base, heatmap, alpha)

                # 잎 마스크로 집중도 계산
                # 잎 분리에 실패하면 집중도가 무조건 1.0이 되므로,
                # 그런 경우는 결측(NaN)으로 남겨 평균에서 제외합니다.
                bgr = cv2.cvtColor((base * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                mask, mask_ok = leaf_mask_with_status(bgr)
                focus = lesion_focus_score(heatmap, mask) if mask_ok else float("nan")

                correct = (pred_index == int(row["label_index"]))
                records.append({
                    "source": source_key,
                    "path": str(path),
                    "true_label": cls,
                    "pred_label": classes[pred_index],
                    "correct": correct,
                    "confidence": confidence,
                    "leaf_focus": focus,
                    "mask_ok": mask_ok,
                })

                # 그림 저장
                fig, axes = plt.subplots(1, 3, figsize=(11, 4))
                axes[0].imshow(base)
                axes[0].set_title("Original", fontsize=10)
                axes[1].imshow(heatmap, cmap="jet")
                axes[1].set_title("Grad-CAM", fontsize=10)
                axes[2].imshow(blended)
                axes[2].set_title("Overlay", fontsize=10)
                for ax in axes:
                    ax.axis("off")
                status = "OK" if correct else "WRONG"
                focus_text = f"{focus:.2f}" if mask_ok else "N/A (mask failed)"
                fig.suptitle(
                    f"[{source_key}] true={cls} / pred={classes[pred_index]} "
                    f"({confidence:.2f}) {status} | leaf-focus={focus_text}",
                    fontsize=10,
                )
                fig.tight_layout()
                target = out_dir / source_key / cls
                target.mkdir(parents=True, exist_ok=True)
                fig.savefig(target / f"{i:02d}_{path.stem}.png", dpi=110,
                            bbox_inches="tight")
                plt.close(fig)
    finally:
        cam.close()

    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default=None,
        help="run_2_train.py 때와 같은 --model 값을 주면 그 architecture의 "
             "결과(results/<모델이름>/)로 Grad-CAM을 만듭니다. 생략하면 config.yaml 값을 씁니다.",
    )
    parser.add_argument(
        "--tag", default=None,
        help="같은 모델로 조건만 바꿔 여러 번 돌릴 때 결과를 results/<모델이름>_<tag>/ 에 따로 저장합니다.",
    )
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    device = get_device()
    classes = cfg["classes"]
    manifest_dir = Path(cfg["output"]["root"]) / "manifests"
    out_root = ensure_dir(apply_model_override(cfg, args.model, args.tag))
    cam_dir = ensure_dir(out_root / "gradcam")

    ckpt = out_root / "checkpoints" / "best.pt"
    model = load_checkpoint(ckpt, cfg, len(classes), device)

    print("=" * 62)
    print("  [4단계] Grad-CAM — 판단 근거 시각화")
    print("=" * 62)

    all_records = []

    a_manifest = manifest_dir / "source_A.csv"
    if a_manifest.exists():
        a_frame = pd.read_csv(a_manifest)
        a_test = a_frame[a_frame["split"] == "test"]
        if len(a_test):
            print(f"  source_A 시험셋 처리 중... ({len(a_test)}장 중 일부)")
            all_records += run_source(model, a_test, cfg, device, classes,
                                      "source_A", cam_dir)

    b_manifest = manifest_dir / "source_B.csv"
    if b_manifest.exists():
        b_frame = pd.read_csv(b_manifest)
        if len(b_frame):
            print(f"  source_B 처리 중... ({len(b_frame)}장 중 일부)")
            all_records += run_source(model, b_frame, cfg, device, classes,
                                      "source_B", cam_dir)

    if not all_records:
        print("  [건너뜀] 처리할 이미지가 없습니다.")
        return 1

    frame = pd.DataFrame(all_records)
    frame.to_csv(out_root / "04_gradcam_records.csv", index=False, encoding="utf-8")

    # ---- 잎 집중도 비교 -------------------------------------------------
    print()
    print("-" * 62)
    print("  잎 집중도 (1에 가까울수록 잎을 보고 판단, 0이면 배경을 봄)")
    print("-" * 62)
    print(f"  {'출처':<14}{'평균 집중도':>14}{'정답률':>10}{'유효/전체':>12}")
    summary = {}
    for source, group in frame.groupby("source"):
        valid = group[group["mask_ok"]]
        acc = float(group["correct"].mean())
        n_valid, n_total = int(len(valid)), int(len(group))
        focus = float(valid["leaf_focus"].mean()) if n_valid else float("nan")
        summary[source] = {
            "mean_leaf_focus": None if n_valid == 0 else focus,
            "accuracy": acc,
            "n": n_total,
            "n_valid_mask": n_valid,
        }
        focus_text = f"{focus:>14.3f}" if n_valid else f"{'측정불가':>14}"
        print(f"  {source:<14}{focus_text}{acc * 100:>9.1f}%"
              f"{f'{n_valid}/{n_total}':>12}")

    # 잎 분리가 많이 실패했으면 경고합니다.
    failed = frame[~frame["mask_ok"]]
    if len(failed):
        print()
        print(f"  [주의] 잎 분리에 실패한 이미지가 {len(failed)}장 있어 평균에서 제외했습니다.")
        print("         배경 색이 잎과 비슷하면 실패합니다. 실패가 절반을 넘으면")
        print("         이 집중도 수치는 논문에 쓰지 마세요.")
        for source, group in failed.groupby("source"):
            print(f"           {source}: {len(group)}장 실패")

    a_ok = summary.get("source_A", {}).get("mean_leaf_focus") is not None
    b_ok = summary.get("source_B", {}).get("mean_leaf_focus") is not None
    if a_ok and b_ok:
        diff = summary["source_A"]["mean_leaf_focus"] - summary["source_B"]["mean_leaf_focus"]
        print("  " + "-" * 58)
        print(f"  집중도 차이 (A - B): {diff:+.3f}")
        print()
        if diff > 0.15:
            print("  해석: 새 이미지에서 모델의 시선이 잎 밖으로 흩어졌습니다.")
            print("        성능 저하의 원인이 '배경 의존'일 가능성이 큽니다.")
        elif diff < -0.15:
            print("  해석: 새 이미지에서 오히려 잎에 더 집중했습니다.")
            print("        성능이 떨어졌다면 배경이 아닌 다른 원인을 찾아야 합니다.")
        else:
            print("  해석: 두 출처에서 시선 분포가 비슷합니다.")
            print("        성능 차이는 배경보다 병반 자체의 생김새 차이일 수 있습니다.")
    else:
        print("  " + "-" * 58)
        print("  두 출처를 비교하기에 유효한 표본이 부족합니다.")

    # 틀린 사례 중 집중도가 낮은 것 = 가장 좋은 논문 그림 후보
    wrong = frame[(~frame["correct"]) & frame["mask_ok"]].sort_values("leaf_focus")
    if len(wrong):
        print()
        print("-" * 62)
        print("  논문에 넣기 좋은 실패 사례 (틀렸고 배경을 본 경우)")
        for _, row in wrong.head(5).iterrows():
            print(f"    [{row['source']}] {row['true_label']} -> {row['pred_label']}"
                  f"  집중도 {row['leaf_focus']:.2f}")
            print(f"      {row['path']}")

    save_json(summary, out_root / "04_gradcam_summary.json")

    print()
    print("=" * 62)
    print(f"  히트맵 이미지: {cam_dir}")
    print("  4단계 완료. 이제 run_5_report.py 를 실행하세요.")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
