"""
데이터셋 준비 도구 — 내려받은 감귤 데이터를 파이프라인이 읽는 구조로 바꾸고,
두 데이터셋이 같은 원본에서 나온 건 아닌지 검사합니다.

필요 패키지: pillow 만 있으면 됩니다.

─────────────────────────────────────────────────────────────────────────
1) 폴더 구조 살펴보기
   py prepare_dataset.py inspect  "C:\\down\\citrus"

2) "작물,병해" 구조로 변환 (원본은 건드리지 않고 복사)
   py prepare_dataset.py convert "C:\\down\\citrus" "data\\citrusA" --crop "Citrus unshiu"

3) 두 데이터셋 중복 검사  ← 교차검증 하려면 반드시
   py prepare_dataset.py dedup "data\\citrusA" "data\\citrusB"
─────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import sys, shutil, argparse, re, json
from pathlib import Path
from collections import Counter, defaultdict

try:
    from PIL import Image
except ImportError:
    sys.exit("pillow 가 필요합니다:  py -m pip install pillow")

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# 흔한 폴더 이름 → 표준 병해명. 필요하면 여기에 추가하세요.
DISEASE_ALIASES = {
    'healthy': 'Healthy', 'health': 'Healthy', 'normal': 'Healthy',
    'fresh': 'Healthy', 'nodisease': 'Healthy', 'good': 'Healthy',
    'blackspot': 'Black Spot', 'black_spot': 'Black Spot',
    'canker': 'Canker', 'cankers': 'Canker',
    'greening': 'Greening', 'huanglongbing': 'Greening', 'hlb': 'Greening',
    'citrusgreening': 'Greening',
    'melanose': 'Melanose', 'scab': 'Scab', 'scabs': 'Scab',
    'greasyspot': 'Greasy Spot', 'leafminer': 'Leaf Miner',
    'citrusleafminer': 'Leaf Miner', 'minerleaf': 'Leaf Miner',
    'anthracnose': 'Anthracnose', 'mite': 'Mite', 'redscale': 'Red Scale',
    'nutrientdeficiency': 'Nutrient Deficiency',
    'irondeficiency': 'Iron Deficiency',
    'magnesiumdeficiency': 'Magnesium Deficiency',
    'manganesedeficiency': 'Manganese Deficiency',
    'nitrogendeficiency': 'Nitrogen Deficiency',
    'zincdeficiency': 'Zinc Deficiency',
}


def norm_key(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', s.lower())


def pretty(s: str) -> str:
    """'black_spot' / 'BlackSpot' → 'Black Spot'"""
    s = re.sub(r'[_\-]+', ' ', s)
    s = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', s)
    return ' '.join(w.capitalize() for w in s.split())


# 병해명에 붙어도 의미 없는 꼬리말
_NOISE = ('leaf', 'leaves', 'disease', 'diseased', 'images', 'image',
          'citrus', 'plant', 'infected', 'class', 'train', 'test')

def map_disease(folder_name: str) -> str:
    raw = folder_name.split('___')[-1]          # PlantVillage 스타일
    raw = raw.split(',')[-1]

    # 괄호 안 약어를 본체와 함께 후보로 삼음:  "Greening(HLB)" → greening / hlb
    inner = re.findall(r'[\(\[]([^)\]]+)[\)\]]', raw)
    outer = re.sub(r'[\(\[][^)\]]*[\)\]]', ' ', raw)
    for cand in [outer] + inner:
        k = norm_key(cand)
        if k in DISEASE_ALIASES:
            return DISEASE_ALIASES[k]

    # 꼬리말 제거 후 재시도:  "Melanose_leaf" → melanose
    words = [w for w in re.split(r'[^A-Za-z0-9]+', outer) if w]
    kept  = [w for w in words if norm_key(w) not in _NOISE] or words
    k = norm_key(''.join(kept))
    if k in DISEASE_ALIASES:
        return DISEASE_ALIASES[k]
    return pretty(' '.join(kept)) if kept else pretty(raw)


def images_in(d: Path):
    return [f for f in d.rglob('*') if f.suffix.lower() in IMG_EXT and f.is_file()]


# ══════════════════════════════════════════════════════════════════════════
def cmd_inspect(args):
    root = Path(args.src)
    if not root.exists():
        sys.exit(f"경로 없음: {root}")
    print(f"\n[{root}]\n")
    subs = [d for d in sorted(root.iterdir()) if d.is_dir()]
    if not subs:
        print(f"  하위 폴더 없음 — 이미지 {len(images_in(root))}장이 바로 들어있습니다.")
        print("  → 클래스별 폴더로 나뉜 상위 폴더를 지정하세요.")
        return
    total = 0
    print(f"  {'하위 폴더':<45} {'이미지':>7}   → 변환될 병해명")
    print("  " + "-" * 82)
    for d in subs:
        n = len(images_in(d)); total += n
        print(f"  {d.name[:44]:<45} {n:>7}   → {map_disease(d.name)}")
    print("  " + "-" * 82)
    print(f"  {'합계':<45} {total:>7}   ({len(subs)}개 클래스)\n")
    print("  병해명이 이상하면 prepare_dataset.py 의 DISEASE_ALIASES 에 추가하거나,")
    print("  convert 실행 후 폴더 이름을 직접 고치면 됩니다.\n")


# ══════════════════════════════════════════════════════════════════════════
def cmd_convert(args):
    src, dst = Path(args.src), Path(args.dst)
    if not src.exists():
        sys.exit(f"경로 없음: {src}")
    crop = args.crop
    subs = [d for d in sorted(src.iterdir()) if d.is_dir()]
    if not subs:
        sys.exit("하위 폴더가 없습니다. inspect 로 구조를 먼저 확인하세요.")

    dst.mkdir(parents=True, exist_ok=True)
    manifest, total = [], 0
    for d in subs:
        disease = map_disease(d.name)
        out = dst / f"{crop},{disease}"
        out.mkdir(exist_ok=True)
        imgs = images_in(d)
        if args.limit:
            imgs = imgs[:args.limit]
        for i, f in enumerate(imgs):
            target = out / f"{norm_key(d.name)}_{i:05d}{f.suffix.lower()}"
            if not target.exists():
                shutil.copy2(f, target)
            manifest.append({'src': str(f), 'dst': str(target),
                             'class': f"{crop},{disease}"})
        total += len(imgs)
        print(f"  {d.name[:40]:<42} → {crop},{disease:<28} {len(imgs):>6}장")

    (dst / '_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f"\n완료: {total}장 → {dst}")
    print(f"출처 추적용 목록: {dst / '_manifest.json'}")
    print(f"\n웹 UI 의 '데이터셋 폴더' 칸에 이 경로를 넣으세요:\n  {dst}\n")


# ══════════════════════════════════════════════════════════════════════════
def ahash(path, size=8):
    """average hash — 크기·압축이 달라도 같은 사진이면 같은 값이 나옵니다."""
    try:
        im = Image.open(path).convert('L').resize((size, size), Image.LANCZOS)
    except Exception:
        return None
    px = list(im.tobytes())
    avg = sum(px) / len(px)
    bits = 0
    for i, v in enumerate(px):
        if v > avg:
            bits |= (1 << i)
    return bits


def hamming(a, b):
    return bin(a ^ b).count('1')


def cmd_dedup(args):
    A, B = Path(args.a), Path(args.b)
    for p in (A, B):
        if not p.exists(): sys.exit(f"경로 없음: {p}")

    print("\n해시 계산 중… (장수에 따라 수십 초 걸릴 수 있습니다)")
    ha, hb = {}, {}
    for label, root, store in (('A', A, ha), ('B', B, hb)):
        imgs = images_in(root)
        for i, f in enumerate(imgs):
            h = ahash(f)
            if h is not None:
                store.setdefault(h, []).append(f)
            if (i + 1) % 200 == 0:
                print(f"  {label}: {i+1}/{len(imgs)}")
        print(f"  {label}: {len(imgs)}장, 고유 해시 {len(store)}개")

    exact = set(ha) & set(hb)
    n_exact = sum(len(ha[h]) for h in exact)

    # 근접 중복 (해밍거리 <= threshold)
    thr = args.threshold
    near = []
    if thr > 0:
        bkeys = list(hb)
        for h in ha:
            if h in exact: continue
            for hb_k in bkeys:
                if hamming(h, hb_k) <= thr:
                    near.append((ha[h][0], hb[hb_k][0]))
                    break

    print("\n" + "=" * 70)
    print(f"  중복 검사 결과")
    print("=" * 70)
    print(f"  A: {A}")
    print(f"  B: {B}")
    print(f"  완전 동일 이미지 : {n_exact}장 ({len(exact)}개 해시)")
    print(f"  근접 중복(거리≤{thr}) : {len(near)}쌍")

    total_a = sum(len(v) for v in ha.values())
    ratio = (n_exact + len(near)) / max(total_a, 1) * 100
    print(f"  A 기준 중복 비율  : {ratio:.1f}%")
    print()
    if ratio >= 5:
        print("  ⛔ 두 데이터셋이 같은 원본에서 파생됐을 가능성이 높습니다.")
        print("     이 상태로 교차검증하면 '다른 출처' 조건이 깨져 결과가 무효가 됩니다.")
        print("     → 다른 데이터셋을 고르거나, 중복분을 제거한 뒤 사용하세요.")
    elif ratio > 0:
        print("  ⚠ 일부 겹칩니다. 겹치는 이미지를 제거하고 쓰는 편이 안전합니다.")
    else:
        print("  ✅ 겹치는 이미지가 없습니다. 서로 다른 출처로 볼 수 있습니다.")
    print()

    if exact or near:
        rows = [{'a': str(ha[h][0]), 'b': str(hb[h][0]), 'type': 'exact'} for h in exact]
        rows += [{'a': str(x), 'b': str(y), 'type': 'near'} for x, y in near]
        out = Path(args.out or 'overlap_report.json')
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f"  겹치는 목록 저장: {out}  ({len(rows)}건)\n")


# ══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="감귤 데이터셋 준비 도구")
    sub = ap.add_subparsers(dest='cmd', required=True)

    p1 = sub.add_parser('inspect', help='폴더 구조와 클래스별 장수 확인')
    p1.add_argument('src')
    p1.set_defaults(func=cmd_inspect)

    p2 = sub.add_parser('convert', help='"작물,병해" 구조로 복사')
    p2.add_argument('src'); p2.add_argument('dst')
    p2.add_argument('--crop', default='Citrus unshiu', help='작물명 (기본: Citrus unshiu)')
    p2.add_argument('--limit', type=int, default=0, help='클래스당 최대 장수 (0=전부)')
    p2.set_defaults(func=cmd_convert)

    p3 = sub.add_parser('dedup', help='두 데이터셋 중복 검사')
    p3.add_argument('a'); p3.add_argument('b')
    p3.add_argument('--threshold', type=int, default=3, help='근접 중복 해밍거리 (기본 3)')
    p3.add_argument('--out', default='overlap_report.json')
    p3.set_defaults(func=cmd_dedup)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
