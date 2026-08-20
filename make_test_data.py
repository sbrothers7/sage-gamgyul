"""
파이프라인 시험용 가짜 이미지 생성기

진짜 감귤 사진을 아직 못 구했을 때, 코드가 제대로 도는지 먼저 확인하려고
만든 도구입니다. 잎 모양에 병반을 찍은 가짜 이미지를 만들어 냅니다.

중요:
  - source_A 와 source_B 를 일부러 다른 조건으로 만듭니다.
    (B는 배경색, 조명, 병반 색이 다름 → 실제 '도메인 이동'을 흉내)
  - 여기서 나오는 숫자는 연구 결과가 아닙니다. 배관 점검용일 뿐입니다.
  - 진짜 데이터를 넣기 전에 반드시 이 가짜 데이터를 지우세요.

실행:
  python make_test_data.py          가짜 데이터 생성
  python make_test_data.py --clean  가짜 데이터 삭제
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parent


def make_leaf(rng, size, background, leaf_color, spot_color, n_spots,
              spot_radius, blur):
    """잎 하나를 그립니다."""
    image = Image.new("RGB", (size, size), background)
    draw = ImageDraw.Draw(image)

    # 잎 모양 (타원 두 개를 겹쳐 뾰족하게)
    cx, cy = size // 2, size // 2
    w = int(size * rng.uniform(0.30, 0.40))
    h = int(size * rng.uniform(0.40, 0.48))
    draw.ellipse([cx - w, cy - h, cx + w, cy + h], fill=leaf_color)

    # 잎맥
    vein = tuple(max(0, c - 35) for c in leaf_color)
    draw.line([cx, cy - h, cx, cy + h], fill=vein, width=max(2, size // 90))
    for i in range(-4, 5):
        y = cy + int(i * h / 5.5)
        dx = int(w * 0.75 * (1 - abs(i) / 6.0))
        draw.line([cx, y, cx - dx, y + int(h * 0.09)], fill=vein, width=1)
        draw.line([cx, y, cx + dx, y + int(h * 0.09)], fill=vein, width=1)

    # 병반
    for _ in range(n_spots):
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0, 0.72)
        sx = cx + int(np.cos(angle) * radius * w)
        sy = cy + int(np.sin(angle) * radius * h)
        r = int(rng.uniform(*spot_radius) * size)
        jitter = tuple(int(np.clip(c + rng.integers(-22, 23), 0, 255))
                       for c in spot_color)
        draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=jitter)

    if blur > 0:
        image = image.filter(ImageFilter.GaussianBlur(blur))
    return image


# (병해이름, 병반 개수 범위, 병반 크기 범위, 병반 색)
DISEASE_STYLE = {
    "healthy": ((0, 1), (0.004, 0.008), (95, 150, 70)),
    "canker": ((9, 18), (0.014, 0.026), (168, 122, 58)),   # 갈색 융기 반점
    "greening": ((4, 8), (0.045, 0.075), (208, 196, 78)),  # 노란 얼룩
}

# 출처별 촬영 조건 차이 — 이것이 '도메인 이동'입니다
SOURCE_STYLE = {
    "source_A": {  # 실험실 조건: 흰 배경, 균일한 조명
        "background": (238, 238, 236),
        "leaf_color": (72, 128, 62),
        "blur": 0.4,
        "size": 224,
        "brightness": 1.0,
    },
    "source_B": {  # 야외 조건: 어두운 배경, 다른 색감, 흐릿함
        "background": (118, 104, 84),
        "leaf_color": (94, 116, 54),
        "blur": 1.3,
        "size": 224,
        "brightness": 0.82,
    },
}


def build(source: str, per_class: int, seed: int) -> int:
    rng = np.random.default_rng(seed)
    style = SOURCE_STYLE[source]
    root = PROJECT_ROOT / "data" / "raw" / source
    made = 0

    for disease, (n_range, r_range, spot_color) in DISEASE_STYLE.items():
        folder = root / disease
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            image = make_leaf(
                rng,
                size=style["size"],
                background=style["background"],
                leaf_color=style["leaf_color"],
                spot_color=spot_color,
                n_spots=int(rng.integers(*n_range)) if n_range[1] > n_range[0] else n_range[0],
                spot_radius=r_range,
                blur=style["blur"],
            )
            if style["brightness"] != 1.0:
                array = np.asarray(image).astype(np.float32) * style["brightness"]
                image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
            image.save(folder / f"{source}_{disease}_{i:03d}.jpg", quality=92)
            made += 1
    return made


def main() -> int:
    parser = argparse.ArgumentParser(description="파이프라인 시험용 가짜 데이터")
    parser.add_argument("--clean", action="store_true", help="가짜 데이터 삭제")
    parser.add_argument("--per-class-a", type=int, default=40,
                        help="source_A 병해별 장수 (기본 40)")
    parser.add_argument("--per-class-b", type=int, default=15,
                        help="source_B 병해별 장수 (기본 15)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw = PROJECT_ROOT / "data" / "raw"
    nobg = PROJECT_ROOT / "data" / "nobg"

    if args.clean:
        removed = 0
        for folder in [raw / "source_A", raw / "source_B", nobg]:
            if folder.exists():
                shutil.rmtree(folder)
                removed += 1
                print(f"  삭제: {folder}")
        (raw / "source_A").mkdir(parents=True, exist_ok=True)
        (raw / "source_B").mkdir(parents=True, exist_ok=True)
        print(f"\n  {removed}개 폴더를 비웠습니다. 이제 진짜 데이터를 넣으세요.")
        return 0

    print("=" * 62)
    print("  파이프라인 시험용 가짜 데이터 생성")
    print("=" * 62)
    print("  ※ 이것은 연구 데이터가 아닙니다. 코드 점검용입니다.")
    print()

    n_a = build("source_A", args.per_class_a, args.seed)
    print(f"  source_A: {n_a}장 생성 (실험실 조건 - 흰 배경, 밝은 조명)")
    n_b = build("source_B", args.per_class_b, args.seed + 1000)
    print(f"  source_B: {n_b}장 생성 (야외 조건 - 어두운 배경, 다른 색감)")

    print()
    print("=" * 62)
    print("  이제 순서대로 실행해서 코드가 도는지 확인하세요:")
    print("    python run_1_prepare.py")
    print("    python run_2_train.py")
    print("    python run_3_evaluate.py")
    print("    python run_4_gradcam.py")
    print("    python run_5_report.py")
    print()
    print("  확인이 끝나면 반드시 지우세요:")
    print("    python make_test_data.py --clean")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
