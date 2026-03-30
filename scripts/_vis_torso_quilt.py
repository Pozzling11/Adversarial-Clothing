"""Visualize torso quilt placement on 5 random training images."""
import numpy as np
import cv2
import random
from ultralytics import YOLO
from pathlib import Path

model = YOLO("yolov8n-pose.pt")
img_dir = Path("data/TRAINING LEG IMAGES _preprocessed")
out_dir = Path("torso_quilt_vis")
out_dir.mkdir(exist_ok=True)

imgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.avif"))
if not imgs:
    imgs = sorted(img_dir.iterdir())

random.shuffle(imgs)
count = 0

for p in imgs:
    if count >= 5:
        break
    img = cv2.imread(str(p))
    if img is None:
        continue
    h0, w0 = img.shape[:2]
    scale = 640 / max(h0, w0)
    img_resized = cv2.resize(img, (int(w0 * scale), int(h0 * scale)))
    ih, iw = img_resized.shape[:2]

    results = model(img, imgsz=640, verbose=False)
    for r in results:
        if r.keypoints is None:
            continue
        kpts = r.keypoints.data.cpu().numpy()
        if len(kpts) == 0:
            continue
        k = kpts[0]  # first person

        ls_x, ls_y = k[5][:2] * scale
        rs_x, rs_y = k[6][:2] * scale
        lh_x, lh_y = k[11][:2] * scale
        rh_x, rh_y = k[12][:2] * scale

        sw = max(ls_x, rs_x) - min(ls_x, rs_x)
        if sw < 10:
            continue
        side = max(4, int(sw))
        margin = sw * 0.20

        # Old bounding box (shoulder-to-shoulder, no margin)
        old_x1 = int(max(0, min(ls_x, rs_x)))
        old_x2 = int(min(iw, max(ls_x, rs_x)))
        old_y1 = int(max(0, min(ls_y, rs_y)))
        old_y2 = int(min(ih, max(lh_y, rh_y)))

        # New bounding box (with 20% margin)
        new_x1 = int(max(0, min(ls_x, rs_x) - margin))
        new_x2 = int(min(iw, max(ls_x, rs_x) + margin))
        new_y1 = old_y1
        new_y2 = old_y2

        vis = img_resized.copy()

        # Draw old box in RED (dashed)
        for i in range(old_x1, old_x2, 8):
            cv2.line(vis, (i, old_y1), (min(i + 4, old_x2), old_y1), (0, 0, 255), 2)
            cv2.line(vis, (i, old_y2), (min(i + 4, old_x2), old_y2), (0, 0, 255), 2)
        for i in range(old_y1, old_y2, 8):
            cv2.line(vis, (old_x1, i), (old_x1, min(i + 4, old_y2)), (0, 0, 255), 2)
            cv2.line(vis, (old_x2, i), (old_x2, min(i + 4, old_y2)), (0, 0, 255), 2)

        # Draw new box in GREEN (solid)
        cv2.rectangle(vis, (new_x1, new_y1), (new_x2, new_y2), (0, 255, 0), 2)

        # Draw tile grid inside new box (cyan lines)
        # Vertical tile lines
        x = new_x1 + side
        while x < new_x2:
            cv2.line(vis, (int(x), new_y1), (int(x), new_y2), (255, 255, 0), 1)
            x += side
        # Horizontal tile lines
        y = new_y1 + side
        while y < new_y2:
            cv2.line(vis, (new_x1, int(y)), (new_x2, int(y)), (255, 255, 0), 1)
            y += side

        # Draw shoulder keypoints (blue circles)
        cv2.circle(vis, (int(ls_x), int(ls_y)), 5, (255, 0, 0), -1)
        cv2.circle(vis, (int(rs_x), int(rs_y)), 5, (255, 0, 0), -1)
        # Draw hip keypoints (magenta circles)
        cv2.circle(vis, (int(lh_x), int(lh_y)), 5, (255, 0, 255), -1)
        cv2.circle(vis, (int(rh_x), int(rh_y)), 5, (255, 0, 255), -1)

        # Legend
        cv2.putText(vis, "RED dashed = old (shoulder-to-shoulder)", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        cv2.putText(vis, "GREEN = new (+20% margin each side)", (5, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(vis, "CYAN grid = tile boundaries", (5, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        # Stats
        old_w = old_x2 - old_x1
        new_w = new_x2 - new_x1
        torso_h = new_y2 - new_y1
        cv2.putText(vis, f"tile={side}px  old_w={old_w}  new_w={new_w}  h={torso_h}  tiles={new_w/side:.1f}x{torso_h/side:.1f}",
                    (5, ih - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        out_path = out_dir / f"torso_quilt_{count}.jpg"
        cv2.imwrite(str(out_path), vis)
        print(f"Saved {out_path}  tile={side}px  old_w={old_w}  new_w={new_w}  h={torso_h}  grid={new_w/side:.1f}x{torso_h/side:.1f}")
        count += 1
        break

print(f"\nDone — {count} images saved to {out_dir}/")
