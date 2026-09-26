"""
데이터 만들기 — 공개 데이터셋을 내려받아 data/raw/source_A, source_B 를 만듭니다.

하는 일:
  1. 네 개의 공개 데이터셋을 data/datasets/<이름>/ 에 내려받습니다.
     (이미 있으면 건너뜁니다.)
  2. 중복·증강본을 걸러서 data/raw/ 에 넣습니다.
       source_A/<병해>/            ← Citrus Leaves (재현용)
       source_B/<병해>/<출처>/     ← PlantVillage, kaku321, CitrusUAT (검증용)

정리 규칙은 docs/01_data_collection_guide.md 의 출처 기록표 아래에 있습니다.
source_B 안에 직접 찍은 사진 폴더(예: source_B/healthy/jeju/)를 만들어 두면
건드리지 않습니다. 이 스크립트가 만드는 출처 폴더만 새로 씁니다.

필요한 것: git, curl, 그리고 `pip install kagglehub`
내려받는 양: 약 2.3GB (CitrusUAT 압축 파일이 1.9GB)

실행:
  python build_data.py
"""

import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
DATASETS = ROOT / "data" / "datasets"
DOWNLOADS = DATASETS / "_downloads"
RAW = ROOT / "data" / "raw"
CLASSES = ("healthy", "canker", "greening")

# 같은 사진을 다시 저장하거나 밝기만 바꾼 source_A 사본. 파일 해시(MD5)로는
# 안 잡혀서 지각 해시 + 눈으로 확인해 골랐습니다 (2026-09-18).
A_NEAR_DUPLICATES = {
    "greening_9.png", "greening_28.png", "greening_45.png",
    "greening_79.png", "greening_188.png",
}


def run(*cmd, cwd=None):
    subprocess.run(cmd, cwd=cwd, check=True)


def sparse_clone(url, paths, dst):
    if not dst.exists():
        run("git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", url, str(dst))
    run("git", "sparse-checkout", "set", *paths, cwd=dst)
    return dst


def copy_all(src_dir, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(src_dir.iterdir()):
        shutil.copy2(f, dst_dir / f.name)


# ---- 1. 내려받기 ------------------------------------------------------

def fetch_citrus_leaves(out):
    repo = sparse_clone(
        "https://github.com/ai-agriculture-circuits-and-systems/citrus_leaves.git",
        [f"leaves/{c}/images" for c in CLASSES], DOWNLOADS / "citrus_leaves")
    for c in CLASSES:
        copy_all(repo / "leaves" / c / "images", out / c)


def fetch_plant_village_orange(out):
    sub = "oranges/huanglongbing_citrus_greening/color/images"
    repo = sparse_clone(
        "https://github.com/ai-agriculture-circuits-and-systems/Plant_Village_Orange.git",
        [sub], DOWNLOADS / "Plant_Village_Orange")
    copy_all(repo / sub, out / "greening")


def fetch_kaku321(out):
    try:
        import kagglehub
    except ImportError:
        sys.exit("  [오류] kagglehub 가 없습니다. 먼저 `pip install kagglehub` 를 실행하세요.")
    src = Path(kagglehub.dataset_download("kaku321/citrus-plant-disease"))
    shutil.copytree(src, out)


def fetch_orange_leaves_hlb_2025(out):
    """Mendeley jgkh2jxbwt. 잎만 잘라낸 preprocessed 사진만 씁니다
    (healthy 와 greening 둘 다 같은 방식으로 처리된 것이라야 배경이 단서가 되지 않습니다)."""
    zip_path = DOWNLOADS / "orange_leaves_hlb_2025.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if not zipfile.is_zipfile(zip_path):
        run("curl", "-L", "--retry", "5", "-C", "-", "-o", str(zip_path),
            "https://data.mendeley.com/public-files/datasets/jgkh2jxbwt/files/"
            "42c44990-bc2e-44f6-8ea8-0618403612b9/file_downloaded")
    folders = {
        "greening": "Symptoms HLB Leaves/preprocessed_images",
        "healthy": "Healty Leaves/preprocessed_images",  # 원본 폴더 이름의 오타 그대로
    }
    with zipfile.ZipFile(zip_path) as z:
        for cls, sub in folders.items():
            (out / cls).mkdir(parents=True, exist_ok=True)
            for name in z.namelist():
                if sub in name and name.lower().endswith((".png", ".jpg", ".jpeg")):
                    (out / cls / Path(name).name).write_bytes(z.read(name))


def fetch_citrus_uat(out):
    zip_path = DOWNLOADS / "CitrusUAT_dataset.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if not zipfile.is_zipfile(zip_path):
        # 1.9GB 라 중간에 끊기기 쉽습니다. -C - 로 끊긴 곳부터 이어받습니다.
        run("curl", "-L", "--retry", "5", "-C", "-", "-o", str(zip_path),
            "https://zenodo.org/api/records/8294078/files/CitrusUAT_dataset.zip/content")
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            file = Path(name).name
            for prefix, cls in (("Healthy_", "healthy"), ("HLB_", "greening")):
                if "/Images/" in name and file.startswith(prefix):
                    (out / cls).mkdir(parents=True, exist_ok=True)
                    (out / cls / file).write_bytes(z.read(name))


FETCHERS = {
    "citrus_leaves": fetch_citrus_leaves,
    "orange_leaves_hlb_2025": fetch_orange_leaves_hlb_2025,
    "Plant_Village_Orange": fetch_plant_village_orange,
    "citrus-plant-disease": fetch_kaku321,
    "CitrusUAT": fetch_citrus_uat,
}


# ---- 2. 정리해서 data/raw 에 넣기 ------------------------------------

def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def first_of_each_block(folder):
    """kaku321 은 사진 1장을 뒤집고·돌려 48~56장으로 불린 뒤 연속 번호로 저장했습니다.
    크기가 같은 연속 번호 묶음이 사진 1장이므로 묶음마다 첫 장만 씁니다."""
    keep, prev = [], None
    for f in sorted(folder.iterdir(), key=lambda p: int(p.stem)):
        size = tuple(sorted(Image.open(f).size))  # 90도 돌린 것도 같은 묶음
        if size != prev:
            keep.append(f)
        prev = size
    return keep


def build():
    seen = set()  # 이미 넣은 파일의 해시. A와 겹치거나 B 안에서 겹치면 뺍니다.

    print("  source_A (Citrus Leaves + Orange Leaves HLB 2025)")
    for c in CLASSES:
        dst = RAW / "source_A" / c
        dst.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in sorted((DATASETS / "citrus_leaves" / c).iterdir()):
            if f.name in A_NEAR_DUPLICATES:
                (dst / f.name).unlink(missing_ok=True)
                continue
            shutil.copy2(f, dst / f.name)
            seen.add(md5(f))
            n += 1
        # 두 번째 출처. healthy 와 greening 만 있습니다 (canker 없음).
        extra = DATASETS / "orange_leaves_hlb_2025" / c
        n_extra = 0
        if extra.is_dir():
            sub = dst / "orange_leaves_hlb_2025"
            shutil.rmtree(sub, ignore_errors=True)
            sub.mkdir(parents=True)
            for f in sorted(extra.iterdir()):
                if f.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    continue
                digest = md5(f)
                if digest in seen:
                    continue
                seen.add(digest)
                shutil.copy2(f, sub / f.name)
                n_extra += 1
        suffix = f" (+ {n_extra} orange_leaves_hlb_2025)" if n_extra else ""
        print(f"     {c:<10}{n + n_extra:>6} 장{suffix}")

    kaku = DATASETS / "citrus-plant-disease"
    groups = [
        ("greening", "PlantVillage", sorted((DATASETS / "Plant_Village_Orange" / "greening").iterdir())),
        ("greening", "kaku321", first_of_each_block(kaku / "Greening")),
        ("greening", "CitrusUAT", sorted((DATASETS / "CitrusUAT" / "greening").iterdir())),
        ("healthy", "kaku321", first_of_each_block(kaku / "Healthy")),
        ("healthy", "CitrusUAT", sorted((DATASETS / "CitrusUAT" / "healthy").iterdir())),
        ("canker", "kaku321", first_of_each_block(kaku / "Canker")),
    ]
    print("  source_B")
    for cls, origin, files in groups:
        dst = RAW / "source_B" / cls / origin
        shutil.rmtree(dst, ignore_errors=True)
        dst.mkdir(parents=True)
        n = 0
        for f in files:
            digest = md5(f)
            if digest in seen:  # PlantVillage 의 _1 사본 같은 똑같은 파일
                continue
            seen.add(digest)
            shutil.copy2(f, dst / f.name)
            n += 1
        print(f"     {cls + '/' + origin:<24}{n:>6} 장  (후보 {len(files)})")


def main() -> int:
    print("=" * 62)
    print("  데이터 만들기")
    print("=" * 62)
    for name, fetch in FETCHERS.items():
        out = DATASETS / name
        if out.exists():
            print(f"  [있음] {name} — 내려받기 건너뜀")
            continue
        print(f"  [받는 중] {name}")
        part = out.with_name(name + ".part")  # 중간에 끊겨도 반쯤 받은 폴더를 '있음'으로 착각하지 않게
        shutil.rmtree(part, ignore_errors=True)
        fetch(part)
        part.rename(out)
    print()
    build()
    print()
    print("  완료. 이제 python run_1_prepare.py 를 실행하세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
