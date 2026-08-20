"""
[3단계] 일반화 성능 검증  ★ 이 연구의 핵심 ★

하는 일:
  같은 모델을 두 가지 데이터에 각각 적용해서 성적을 비교합니다.

    (1) source_A 시험셋  -> 학습에 쓴 것과 같은 출처. "재현 성능"
    (2) source_B 전체    -> 학습에 쓰지 않은 새 감귤 이미지. "일반화 성능"

  두 성적의 차이가 곧 '일반화 격차'이며, 이 연구가 측정하려는 값입니다.
  Mahapatra et al. (2026) 의 intra-dataset / cross-dataset 대조 설계를 따릅니다.

  설정에서 배경 제거를 켜 두었다면, 배경을 지운 조건도 함께 평가합니다.
  배경을 지웠을 때 점수가 크게 변한다면, 모델이 병반이 아니라
  촬영 환경을 학습했다는 증거가 됩니다.

실행:
  python run_3_evaluate.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from src.core import (
    PROJECT_ROOT, compute_metrics, ensure_dir, get_device, load_checkpoint,
    load_config, make_loader, predict, print_metrics, save_json, scan_source,
    set_seed,
)


def evaluate_frame(model, frame, cfg, device, classes, title):
    """주어진 이미지 목록에 대해 성적을 냅니다."""
    if frame is None or len(frame) == 0:
        return None, None
    loader = make_loader(frame, cfg, training=False, shuffle=False)
    y_true, y_pred, y_prob, paths = predict(model, loader, device)
    metrics = compute_metrics(y_true, y_pred, classes)
    print_metrics(title, metrics)
    predictions = pd.DataFrame({
        "path": paths,
        "true_index": y_true,
        "pred_index": y_pred,
        "true_label": [classes[i] for i in y_true],
        "pred_label": [classes[i] for i in y_pred],
        "confidence": [max(p) for p in y_prob],
        "correct": [t == p for t, p in zip(y_true, y_pred)],
    })
    return metrics, predictions


def main() -> int:
    cfg = load_config()
    set_seed(cfg["seed"])
    device = get_device()
    classes = cfg["classes"]
    out_root = ensure_dir(cfg["output"]["root"])
    metrics_dir = ensure_dir(out_root / "metrics")
    manifest_dir = Path(cfg["output"]["root"]) / "manifests"

    ckpt = Path(cfg["output"]["root"]) / "checkpoints" / "best.pt"
    model = load_checkpoint(ckpt, cfg, len(classes), device)

    print("=" * 62)
    print("  [3단계] 일반화 성능 검증")
    print("=" * 62)
    print(f"  모델: {cfg['model']['name']}  ({ckpt})")

    results = {}

    # ---- (1) 재현 성능: source_A 시험셋 -------------------------------
    a_manifest = manifest_dir / "source_A.csv"
    if not a_manifest.exists():
        print("[오류] source_A 목록이 없습니다. run_1_prepare.py 를 먼저 실행하세요.")
        return 1
    a_frame = pd.read_csv(a_manifest)
    a_test = a_frame[a_frame["split"] == "test"]

    m_a, pred_a = evaluate_frame(
        model, a_test, cfg, device, classes,
        f"(1) 재현 성능 — {cfg['data']['source_A_label']} 시험셋",
    )
    if m_a:
        results["A_test"] = m_a
        pred_a.to_csv(metrics_dir / "predictions_A_test.csv",
                      index=False, encoding="utf-8")

    # ---- (2) 일반화 성능: source_B ------------------------------------
    b_manifest = manifest_dir / "source_B.csv"
    if b_manifest.exists():
        b_frame = pd.read_csv(b_manifest)
        m_b, pred_b = evaluate_frame(
            model, b_frame, cfg, device, classes,
            f"(2) 일반화 성능 — {cfg['data']['source_B_label']}",
        )
        if m_b:
            results["B"] = m_b
            pred_b.to_csv(metrics_dir / "predictions_B.csv",
                          index=False, encoding="utf-8")
    else:
        print("\n  [건너뜀] source_B 데이터가 아직 없습니다.")
        print("           새 감귤 이미지를 모은 뒤 run_1_prepare.py 를 다시 돌리세요.")

    # ---- (3) 배경 제거 조건 -------------------------------------------
    if cfg["experiment"]["run_background_removed"]:
        nobg_root = PROJECT_ROOT / "data" / "nobg"
        for key, name in [("A_test_nobg", "source_A"), ("B_nobg", "source_B")]:
            folder = nobg_root / name
            if not folder.exists():
                continue
            scanned = scan_source(folder, classes, name + "_nobg",
                                  cfg["data"]["min_images_per_class"])
            if scanned.total == 0:
                continue
            frame = scanned.frame
            # source_A는 시험셋에 해당하는 파일만 골라냅니다.
            if name == "source_A" and m_a is not None:
                test_stems = {Path(p).stem for p in a_test["path"]}
                frame = frame[frame["path"].apply(lambda p: Path(p).stem in test_stems)]
            if len(frame) == 0:
                continue
            label = ("(3) 배경 제거 — 재현용 시험셋" if name == "source_A"
                     else "(4) 배경 제거 — 신규 감귤 이미지")
            m, _ = evaluate_frame(model, frame, cfg, device, classes, label)
            if m:
                results[key] = m

    # ---- 요약 비교표 ---------------------------------------------------
    print("=" * 62)
    print("  최종 비교")
    print("=" * 62)
    header = f"  {'조건':<26}{'정확도':>9}{'F1':>9}{'MCC':>9}{'장수':>8}"
    print(header)
    print("  " + "-" * 58)

    pretty = {
        "A_test": "재현 (같은 출처)",
        "B": "일반화 (새 이미지)",
        "A_test_nobg": "재현 + 배경제거",
        "B_nobg": "일반화 + 배경제거",
    }
    for key, name in pretty.items():
        if key not in results:
            continue
        m = results[key]
        print(f"  {name:<26}{m['accuracy'] * 100:>8.2f}%"
              f"{m['f1_macro']:>9.3f}{m['mcc']:>9.3f}{m['n_samples']:>8}")

    # 핵심 결론
    if "A_test" in results and "B" in results:
        gap = (results["A_test"]["accuracy"] - results["B"]["accuracy"]) * 100
        f1_gap = results["A_test"]["f1_macro"] - results["B"]["f1_macro"]
        print("  " + "-" * 58)
        print(f"  ** 일반화 격차: 정확도 {gap:+.2f}%p, F1 {f1_gap:+.3f} **")
        print()
        if gap > 20:
            print("  해석: 성능이 크게 떨어졌습니다. 이 모델은 학습에 쓴 데이터")
            print("        바깥에서는 신뢰하기 어렵습니다. 연구 가설이 지지됩니다.")
        elif gap > 5:
            print("  해석: 눈에 띄는 성능 저하가 있습니다. 어떤 병해에서 특히")
            print("        많이 틀렸는지 위의 병해별 표를 확인하세요.")
        elif gap > -5:
            print("  해석: 성능이 대체로 유지되었습니다. 다만 source_B 가 충분히")
            print("        다른 조건에서 촬영된 것이 맞는지 먼저 확인해야 합니다.")
        else:
            print("  해석: 새 데이터에서 오히려 점수가 높습니다. source_B 가 너무")
            print("        쉽거나, A와 B가 겹쳤을 가능성을 점검하세요.")
        results["generalization_gap"] = {
            "accuracy_drop_percentage_points": gap,
            "f1_drop": f1_gap,
        }

    print("=" * 62)

    save_json(results, metrics_dir / "03_all_metrics.json")
    print(f"\n  전체 결과 저장: {metrics_dir / '03_all_metrics.json'}")
    print("  다음: run_4_gradcam.py 로 모델이 무엇을 봤는지 확인하세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
