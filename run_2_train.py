"""
[2단계] 모델 학습 (원 논문 재현)

하는 일:
  - source_A 의 학습 데이터로 모델을 학습시킵니다.
  - 검증 데이터로 가장 좋은 시점의 모델을 저장합니다.
  - 학습 곡선을 기록합니다.

이 단계의 목적은 "원 논문의 성능을 재현하는 것"입니다.
여기서 나온 정확도가 논문이 보고한 수치와 비슷해야
그다음 일반화 검증이 의미를 가집니다.

실행:
  python run_2_train.py
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import torch
import torch.nn as nn

from src.core import (
    apply_model_override, build_model, compute_metrics, ensure_dir,
    get_device, load_config, make_loader, predict, print_metrics,
    save_json, set_seed,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default=None,
        help="timm 모델 이름으로 config.yaml 의 model.name 을 덮어씁니다 "
             "(예: resnet50, densenet121, efficientnet_b0, mobilenetv3_large_100). "
             "생략하면 config.yaml 값을 그대로 씁니다. 결과는 results/<모델이름>/ 에 따로 저장됩니다.",
    )
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    device = get_device()
    classes = cfg["classes"]
    out_root = ensure_dir(apply_model_override(cfg, args.model))
    ckpt_dir = ensure_dir(out_root / "checkpoints")

    manifest = Path(cfg["output"]["root"]) / "manifests" / "source_A.csv"
    if not manifest.exists():
        print("[오류] 데이터 목록이 없습니다. 먼저 run_1_prepare.py 를 실행하세요.")
        return 1

    frame = pd.read_csv(manifest)
    train_frame = frame[frame["split"] == "train"]
    val_frame = frame[frame["split"] == "val"]

    if len(train_frame) == 0:
        print("[오류] 학습용 이미지가 없습니다.")
        return 1

    print("=" * 62)
    print("  [2단계] 모델 학습 — 원 논문 재현")
    print("=" * 62)
    print(f"  모델        : {cfg['model']['name']} (ImageNet 사전학습={cfg['model']['pretrained']})")
    print(f"  연산 장치   : {device}")
    print(f"  학습 이미지 : {len(train_frame)}장")
    print(f"  검증 이미지 : {len(val_frame)}장")
    print(f"  최대 에폭   : {cfg['train']['epochs']}")
    print(f"  배치 크기   : {cfg['train']['batch_size']}")
    print(f"  학습률      : {cfg['train']['learning_rate']}")
    if device.type == "cpu":
        print("\n  ※ GPU가 없어 CPU로 학습합니다. 시간이 오래 걸릴 수 있습니다.")
        print("     느리면 config.yaml 의 epochs 를 줄여 보세요.")
    print()

    train_loader = make_loader(train_frame, cfg, training=True)
    val_loader = make_loader(val_frame, cfg, training=False) if len(val_frame) else None

    model = build_model(cfg, len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["epochs"]
    )

    best_score = -1.0
    best_epoch = -1
    patience = cfg["train"]["early_stop_patience"]
    stale = 0
    history = []
    started = time.time()

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        # ---- 학습 ----
        model.train()
        running_loss, seen, correct = 0.0, 0, 0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += labels.size(0)

        train_loss = running_loss / max(seen, 1)
        train_acc = correct / max(seen, 1)

        # ---- 검증 ----
        if val_loader is not None:
            y_true, y_pred, _, _ = predict(model, val_loader, device)
            val_metrics = compute_metrics(y_true, y_pred, classes)
            val_acc = val_metrics["accuracy"]
            val_f1 = val_metrics["f1_macro"]
            score = val_f1
        else:
            val_acc, val_f1, score = float("nan"), float("nan"), train_acc

        scheduler.step()
        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_acc": val_acc, "val_f1": val_f1,
            "lr": optimizer.param_groups[0]["lr"],
        })

        marker = ""
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "classes": classes,
                "model_name": cfg["model"]["name"],
                "val_f1": val_f1,
                "val_acc": val_acc,
            }, ckpt_dir / "best.pt")
            marker = "  <- 최고 기록 (저장됨)"
        else:
            stale += 1

        print(f"  에폭 {epoch:>3}/{cfg['train']['epochs']}  "
              f"손실 {train_loss:.4f}  학습정확도 {train_acc * 100:5.1f}%  "
              f"검증정확도 {val_acc * 100:5.1f}%  검증F1 {val_f1:.3f}{marker}")

        if stale >= patience:
            print(f"\n  검증 성능이 {patience}회 연속 개선되지 않아 조기 종료합니다.")
            break

    elapsed = time.time() - started
    print()
    print("-" * 62)
    print(f"  학습 완료 — {elapsed / 60:.1f}분 소요")
    print(f"  최고 기록: {best_epoch}에폭, 검증 F1 = {best_score:.4f}")
    print(f"  저장 위치: {ckpt_dir / 'best.pt'}")

    pd.DataFrame(history).to_csv(
        out_root / "02_train_history.csv", index=False, encoding="utf-8"
    )
    save_json({
        "model": cfg["model"]["name"],
        "best_epoch": best_epoch,
        "best_val_f1": best_score,
        "elapsed_minutes": elapsed / 60,
        "n_train": len(train_frame),
        "n_val": len(val_frame),
        "config": cfg,
    }, out_root / "02_train_summary.json")

    print()
    print("=" * 62)
    print("  2단계 완료. 이제 run_3_evaluate.py 를 실행하세요.")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
