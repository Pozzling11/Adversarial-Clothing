"""Quick measurement of torso dimensions across training images."""
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import cv2

model = YOLO("yolov8n-pose.pt")
img_dir = Path("data/TRAINING LEG IMAGES _preprocessed")
imgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.avif"))
if not imgs:
    imgs = sorted(img_dir.iterdir())

print(f"{'image':45s}  {'person':8s}  {'sh_w':>6s}  {'torso_h':>7s}  {'torso_w':>7s}  {'tiles_y':>7s}  {'area%':>6s}")
print("-" * 100)

sws, ths, tws = [], [], []
for p in imgs[:20]:
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
        for ki, k in enumerate(kpts):
            ls_x, ls_y = k[5][:2] * scale
            rs_x, rs_y = k[6][:2] * scale
            lh_x, lh_y = k[11][:2] * scale
            rh_x, rh_y = k[12][:2] * scale
            sw = abs(ls_x - rs_x)
            torso_h = max(lh_y, rh_y) - min(ls_y, rs_y)
            torso_w = max(ls_x, rs_x) - min(ls_x, rs_x)
            if sw < 5:
                continue
            sws.append(sw)
            ths.append(torso_h)
            tws.append(torso_w)
            tiles_y = torso_h / sw if sw > 0 else 0
            area_pct = torso_w * torso_h / (640 * 640) * 100
            print(f"{p.name:45s}  person{ki:<2d}  {sw:6.1f}  {torso_h:7.1f}  {torso_w:7.1f}  {tiles_y:7.1f}  {area_pct:5.1f}%")

print("-" * 100)
print(f"{'MEAN':45s}  {'':8s}  {np.mean(sws):6.1f}  {np.mean(ths):7.1f}  {np.mean(tws):7.1f}  {np.mean(ths)/np.mean(sws):7.1f}  {np.mean(tws)*np.mean(ths)/(640*640)*100:5.1f}%")
print(f"{'MIN':45s}  {'':8s}  {np.min(sws):6.1f}  {np.min(ths):7.1f}  {np.min(tws):7.1f}")
print(f"{'MAX':45s}  {'':8s}  {np.max(sws):6.1f}  {np.max(ths):7.1f}  {np.max(tws):7.1f}")

# YOLO stride analysis
print("\n--- YOLO grid cell coverage at far EOT scale (0.20x) ---")
far_scale = 0.20
for label, val in [("shoulder_w", np.mean(sws)), ("torso_h", np.mean(ths)), ("torso_w", np.mean(tws))]:
    px = val * far_scale
    print(f"  {label}: {px:.0f}px -> stride-8: {px/8:.1f} cells, stride-16: {px/16:.1f} cells, stride-32: {px/32:.1f} cells")
