"""
Preview torso-width patch placement on 3 sample host images.
Saves annotated JPGs to torso_width_preview/.

Usage:
    python scripts/preview_torso_placement.py
"""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

IMG_SIZE = 640
DATA_DIR = Path("data/clean")
OUT_DIR  = Path("torso_width_preview")
OUT_DIR.mkdir(exist_ok=True)

yolo = YOLO("yolov8n.pt")

all_imgs = sorted([
    p for p in DATA_DIR.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
])
picks = [all_imgs[0], all_imgs[len(all_imgs) // 2], all_imgs[-1]]

for i, img_path in enumerate(picks, 1):
    raw = cv2.imread(str(img_path))
    if raw is None:
        print(f"skip {img_path.name}")
        continue
    img = cv2.resize(raw, (IMG_SIZE, IMG_SIZE))

    res = yolo.predict(source=img, conf=0.25, classes=[0], verbose=False)
    boxes = res[0].boxes
    if boxes is None or len(boxes) == 0:
        print(f"{img_path.name}: no detection")
        continue
    best = int(boxes.conf.cpu().argmax())
    x1, y1, x2, y2 = boxes.xyxy[best].cpu().numpy().astype(int)

    bbox_w = max(x2 - x1, 1)
    bbox_h = max(y2 - y1, 1)
    # ~55% of bbox_w ≈ actual torso (bbox includes arms), anchor at mid-chest (48% down)
    comp_size = max(32, min(int(bbox_w * 1.0 * 0.55), IMG_SIZE))
    cx  = (x1 + x2) // 2
    cy  = y1 + int(bbox_h * 0.48)   # mid-chest anchor
    row = max(0, min(cy - comp_size // 2, IMG_SIZE - comp_size))
    col = max(0, min(cx - comp_size // 2, IMG_SIZE - comp_size))

    vis = img.copy()
    # Blue: person bbox
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 100, 0), 2)
    # Red semi-transparent fill: patch region
    overlay = vis.copy()
    cv2.rectangle(overlay, (col, row), (col + comp_size, row + comp_size), (0, 0, 220), -1)
    cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)
    cv2.rectangle(vis, (col, row), (col + comp_size, row + comp_size), (0, 0, 255), 2)

    cv2.putText(vis, f"patch {comp_size}x{comp_size}  (55% of bbox_w={bbox_w})",
                (col, max(row - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(vis, img_path.name,
                (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    out = OUT_DIR / f"preview_{i}_{img_path.stem}.jpg"
    cv2.imwrite(str(out), vis)
    print(f"Saved {out}  (patch {comp_size}px, col={col}, row={row})")

print("Done — check torso_width_preview/")
