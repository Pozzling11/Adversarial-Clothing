import random
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import os

# Directory containing images to visualize
IMG_DIR = Path("data/clean")
OUT_DIR = Path("leg_pose_vis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# List all image files in IMG_DIR
image_files = [f for f in IMG_DIR.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.avif'}]

# Select 4 random images
random.seed(42)
selected = random.sample(image_files, min(4, len(image_files)))

# Load YOLOv8-pose model
pose_model = YOLO('yolov8n-pose.pt')

for idx, img_path in enumerate(selected):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] Could not load {img_path}")
        continue
    h0, w0 = img.shape[:2]
    img_resized = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
    results = pose_model(img_resized)
    kpts = results[0].keypoints.data.cpu().numpy() if results[0].keypoints is not None else []
    # Scale keypoints back to original image size
    if len(kpts) > 0:
        scale_x = w0 / 640
        scale_y = h0 / 640
        kpts[:, :, 0] *= scale_x
        kpts[:, :, 1] *= scale_y
        person_kpts = kpts[0]  # first detected person
        hip_idx, ankle_idx = 12, 16
        x_hip, y_hip, conf_hip = person_kpts[hip_idx]
        x_ankle, y_ankle, conf_ankle = person_kpts[ankle_idx]
        if conf_hip < 0.3 or conf_ankle < 0.3:
            print(f"[WARN] Low confidence for hip/ankle in {img_path.name}")
            continue
        hip = np.array([x_hip, y_hip], dtype=np.float32)
        ankle = np.array([x_ankle, y_ankle], dtype=np.float32)
        leg_length = np.linalg.norm(ankle - hip)
        aspect_ratio = 0.25  # width = aspect_ratio * height (e.g., 4:1)
        rect_height = int(leg_length)
        rect_width = int(leg_length * aspect_ratio)
        center = (hip + ankle) / 2
        angle = np.degrees(np.arctan2(ankle[1] - hip[1], ankle[0] - hip[0]))
        box = cv2.boxPoints(((center[0], center[1]), (rect_height, rect_width), angle))
        box = box.astype(np.int32)
        # Draw rectangle overlay on the image
        cv2.drawContours(img, [box], 0, (0, 0, 255), 3)
        # Mark hip and ankle points
        cv2.circle(img, (int(hip[0]), int(hip[1])), 6, (255, 0, 0), -1)
        cv2.circle(img, (int(ankle[0]), int(ankle[1])), 6, (0, 255, 0), -1)
        out_path = OUT_DIR / f"leg_rect_vis_{idx}_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), img)
        print(f"[INFO] Saved {out_path}")
    else:
        print(f"[WARN] No keypoints detected in {img_path.name}")
