"""합성 테스트 데이터셋 생성 — 파이프라인 검증 전용 (과학적 의미 없음)."""
import random, pathlib
from PIL import Image, ImageDraw

CLASSES = [
    "Apple,Black Rot", "Apple,Healthy", "Apple,Scab",
    "Tomato,Early Blight", "Tomato,Healthy", "Tomato,Late Blight",
    "Tomato,Leaf Mold", "Potato,Early Blight", "Potato,Healthy",
    "Corn,Leaf Rust", "Corn,Healthy", "Grape,Black Rot",
    "Grape,Healthy", "Rice,Blast", "Wheat,Healthy",
]
ROOT = pathlib.Path(__file__).resolve().parent / "fake_data" / "images"
random.seed(0)

for cl in CLASSES:
    d = ROOT / cl
    d.mkdir(parents=True, exist_ok=True)
    healthy = cl.endswith("Healthy")
    for i in range(12):
        img = Image.new("RGB", (256, 256), (30, 90, 40))
        dr = ImageDraw.Draw(img)
        dr.ellipse([28, 18, 228, 238], fill=(60, 140, 60))          # 잎
        dr.line([128, 20, 128, 236], fill=(40, 110, 45), width=4)   # 주맥
        if not healthy:                                              # 병반
            for _ in range(random.randint(6, 18)):
                x, y = random.randint(45, 205), random.randint(35, 220)
                r = random.randint(4, 14)
                dr.ellipse([x - r, y - r, x + r, y + r],
                           fill=(random.randint(90, 150), random.randint(60, 95), 30))
        img.save(d / f"{cl.replace(',', '_').replace(' ', '')}_{i:03d}.jpg", quality=85)

n = sum(1 for _ in ROOT.rglob("*.jpg"))
print(f"합성 이미지 {n}장 / {len(CLASSES)}클래스 생성 → {ROOT}")
