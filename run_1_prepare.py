"""
[1단계] 데이터 준비

하는 일:
  - data/raw/source_A 와 source_B 폴더를 훑어서 이미지 목록을 만듭니다.
  - source_A 를 학습/검증/시험으로 나눕니다 (병해별 비율 유지).
  - 데이터에 문제가 없는지 검사해서 알려줍니다.
  - (설정에 따라) 배경을 제거한 사본을 만듭니다.

실행:
  python run_1_prepare.py
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from src.core import (
    PROJECT_ROOT, describe_environment, ensure_dir, load_config,
    save_json, scan_source, set_seed, stratified_split,
)


def main() -> int:
    cfg = load_config()
    set_seed(cfg["seed"])
    classes = cfg["classes"]
    out_root = ensure_dir(cfg["output"]["root"])
    manifest_dir = ensure_dir(out_root / "manifests")

    print("=" * 62)
    print("  [1단계] 데이터 준비")
    print("=" * 62)

    env = describe_environment()
    print(f"  Python {env['python']} / PyTorch {env['torch']}")
    print(f"  연산 장치: {env['gpu']}")
    print(f"  분류 대상 병해 {len(classes)}종: {', '.join(classes)}")
    print()

    problems = []

    # ---- source_A (재현용) -------------------------------------------
    print("-" * 62)
    print(f"  A. {cfg['data']['source_A_label']}")
    print(f"     경로: {cfg['data']['source_A']}")
    a = scan_source(cfg["data"]["source_A"], classes, "source_A",
                    cfg["data"]["min_images_per_class"])

    if a.total == 0:
        print("     [문제] 이미지가 한 장도 없습니다.")
        problems.append(
            "source_A 폴더가 비어 있습니다. "
            "docs/01_data_collection_guide.md 를 보고 데이터를 넣으세요."
        )
    else:
        counts = a.frame["label"].value_counts()
        for cls in classes:
            n = int(counts.get(cls, 0))
            flag = "  <- 없음!" if n == 0 else ("  <- 너무 적음" if n < cfg["data"]["min_images_per_class"] else "")
            print(f"       {cls:<14} {n:>6} 장{flag}")
        print(f"       {'합계':<14} {a.total:>6} 장")

    if a.missing_classes:
        problems.append(f"source_A 에 이미지가 없는 병해: {', '.join(a.missing_classes)}")

    # ---- source_B (일반화 검증용) ------------------------------------
    print()
    print("-" * 62)
    print(f"  B. {cfg['data']['source_B_label']}")
    print(f"     경로: {cfg['data']['source_B']}")
    b = scan_source(cfg["data"]["source_B"], classes, "source_B",
                    cfg["data"]["min_images_per_class"])

    if b.total == 0:
        print("     [비어 있음] 아직 새 감귤 이미지를 모으지 않았습니다.")
        print("                 1~2단계는 이대로 진행할 수 있습니다.")
        print("                 3단계(일반화 검증) 전까지 채우면 됩니다.")
    else:
        counts = b.frame["label"].value_counts()
        for cls in classes:
            n = int(counts.get(cls, 0))
            flag = "  <- 없음!" if n == 0 else ""
            print(f"       {cls:<14} {n:>6} 장{flag}")
        print(f"       {'합계':<14} {b.total:>6} 장")

    # ---- 중복 검사 (데이터 누출 방지) --------------------------------
    # A와 B에 같은 파일이 들어가면 '일반화 검증'이 성립하지 않습니다.
    # (Kapoor & Narayanan, 2023 의 data leakage 점검)
    # 같은 출처 안의 중복도 봅니다. A 안의 사본은 학습/시험에 갈라져 들어가
    # 누출이 되고, B 안의 사본은 장수를 부풀립니다.
    # ※ 바이트가 똑같은 사본만 잡습니다. 크기만 바꾼 사본은 못 잡습니다.
    if a.total or b.total:
        import hashlib
        from collections import Counter

        def digest(path):
            h = hashlib.md5()
            with open(path, "rb") as f:
                h.update(f.read())
            return h.hexdigest()

        print()
        print("-" * 62)
        print("  중복 검사 (데이터 누출 방지)")
        a_digests = [digest(p) for p in a.frame["path"]] if a.total else []
        b_digests = [digest(p) for p in b.frame["path"]] if b.total else []
        for name, digests in [("source_A", a_digests), ("source_B", b_digests)]:
            extra = sum(n - 1 for n in Counter(digests).values() if n > 1)
            if extra:
                print(f"     [경고] {name} 안에 똑같은 이미지가 {extra}장 더 있습니다.")
                problems.append(f"{name} 내부 중복 이미지 {extra}장 — 제거할 것")
        a_hashes = set(a_digests)
        overlap = [p for p, d in zip(b.frame["path"], b_digests) if d in a_hashes] if b.total else []
        if overlap:
            print(f"     [경고] A와 B에 똑같은 이미지가 {len(overlap)}장 있습니다.")
            print("            이러면 일반화 검증이 성립하지 않습니다. 반드시 지우세요.")
            for p in overlap[:5]:
                print(f"              {p}")
            problems.append(f"A/B 중복 이미지 {len(overlap)}장 — 반드시 제거할 것")
        elif not any("중복" in p for p in problems):
            print("     통과 — 겹치는 이미지 없음")

    # ---- 분할 및 저장 -------------------------------------------------
    if a.total > 0:
        split_frame = stratified_split(a.frame, cfg["data"]["split"], cfg["seed"])
        print()
        print("-" * 62)
        print("  source_A 분할 결과")
        pivot = split_frame.pivot_table(
            index="label", columns="split", values="path",
            aggfunc="count", fill_value=0,
        )
        for col in ["train", "val", "test"]:
            if col not in pivot.columns:
                pivot[col] = 0
        pivot = pivot[["train", "val", "test"]]
        print(f"       {'병해':<14}{'학습':>8}{'검증':>8}{'시험':>8}")
        for label, row in pivot.iterrows():
            print(f"       {label:<14}{int(row['train']):>8}{int(row['val']):>8}{int(row['test']):>8}")
        print(f"       {'합계':<14}{int(pivot['train'].sum()):>8}"
              f"{int(pivot['val'].sum()):>8}{int(pivot['test'].sum()):>8}")

        split_frame.to_csv(manifest_dir / "source_A.csv", index=False, encoding="utf-8")
        print(f"\n     저장: {manifest_dir / 'source_A.csv'}")

    if b.total > 0:
        b_frame = b.frame.copy()
        b_frame["split"] = "test"
        b_frame.to_csv(manifest_dir / "source_B.csv", index=False, encoding="utf-8")
        print(f"     저장: {manifest_dir / 'source_B.csv'}")

    # ---- 배경 제거 사본 ------------------------------------------------
    if cfg["experiment"]["run_background_removed"] and (a.total or b.total):
        print()
        print("-" * 62)
        print("  배경 제거 사본 생성 (대조 실험용)")
        print("  ※ 이미지가 많으면 몇 분 걸립니다.")
        from src.explain import process_folder

        for name, source in [("source_A", cfg["data"]["source_A"]),
                             ("source_B", cfg["data"]["source_B"])]:
            dst = PROJECT_ROOT / "data" / "nobg" / name
            shutil.rmtree(dst, ignore_errors=True)  # 지난 실행의 사본이 남지 않게
            if not Path(source).exists():
                continue
            n = process_folder(source, dst)
            print(f"     {name}: {n}장 처리 -> {dst}")

    # ---- 요약 저장 -----------------------------------------------------
    save_json({
        "environment": env,
        "classes": classes,
        "source_A_total": a.total,
        "source_B_total": b.total,
        "problems": problems,
    }, out_root / "01_data_summary.json")

    print()
    print("=" * 62)
    if problems:
        print("  다음 문제를 해결해야 합니다:")
        for i, p in enumerate(problems, 1):
            print(f"    {i}. {p}")
        print("=" * 62)
        return 1

    print("  1단계 완료. 이제 run_2_train.py 를 실행하세요.")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
