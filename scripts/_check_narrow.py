"""Identify narrow-shoulder (side-angle) training images."""
import numpy as np, cv2
from ultralytics import YOLO
from pathlib import Path

model = YOLO("yolov8n-pose.pt")
img_dir = Path("data/TRAINING LEG IMAGES _preprocessed")
imgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.avif"))
if not imgs:
    imgs = sorted(img_dir.iterdir())

narrow = []
all_sws = []
for p in imgs:
    img = cv2.imread(str(p))
    if img is None:
        continue
    h0, w0 = img.shape[:2]
    scale = 640 / max(h0, w0)
    results = model(img, imgsz=640, verbose=False)
    for r in results:
        if r.keypoints is None:
            continue
        kpts = r.keypoints.data.cpu().numpy()
        if len(kpts) == 0:
            continue
        k = kpts[0]
        sw = abs(k[5][0] - k[6][0]) * scale
        all_sws.append(sw)
        flag = "<< NARROW" if sw < 40 else ""
        print(f"{p.name:50s}  sw={sw:6.1f}px  {flag}")
        if sw < 40:
            narrow.append((p.name, sw))

print(f"\nTotal images: {len(all_sws)}")
print(f"Narrow (sw < 40px): {len(narrow)}")
for name, sw in narrow:
    print(f"  {name}: {sw:.1f}px")
print(f"\nShoulder width distribution:")
print(f"  min={min(all_sws):.1f}  median={np.median(all_sws):.1f}  mean={np.mean(all_sws):.1f}  max={max(all_sws):.1f}")
