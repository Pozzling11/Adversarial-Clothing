import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import random

IMG_DIR = Path('data/clean')
OUT_DIR = Path('leg_pose_vis')
OUT_DIR.mkdir(exist_ok=True)

# Load YOLOv8-pose model
yolo_pose = YOLO('yolov8n-pose.pt')

# Get 10 images
img_files = [f for f in IMG_DIR.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']]
img_files = img_files[:10]


for img_path in img_files:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Failed to load {img_path}")
        continue
    h, w = img.shape[:2]
    results = yolo_pose(img, verbose=False)
    if not results or results[0].keypoints is None:
        print(f"No keypoints found for {img_path.name}, skipping.")
        continue
    kpts = results[0].keypoints.data.cpu().numpy()  # (N, 17, 3) for COCO
    for person_id, person_kpts in enumerate(kpts):
        img_kp = img.copy()
        # Draw all keypoints and label them
        for idx, (x, y, conf) in enumerate(person_kpts):
            pt = (int(x), int(y))
            cv2.circle(img_kp, pt, 5, (0, 0, 255), -1)
            cv2.putText(img_kp, str(idx), pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        # COCO: 11=left_ankle, 12=right_ankle, 9=left_knee, 10=right_knee, 5=left_hip, 6=right_hip
            legs = [(12, 14, 16), (12, 14, 16)]  # (HIP, KNEE, ANKLE) for left and right
        leg_idx = random.choice([0, 1])  # Randomly select left or right leg
        # Check confidence for HIP, KNEE, ANKLE
        hip_idx, knee_idx, ankle_idx = legs[leg_idx]
        hip_conf = person_kpts[hip_idx][2]
        knee_conf = person_kpts[knee_idx][2]
        ankle_conf = person_kpts[ankle_idx][2]
        if hip_conf > 0.3 and knee_conf > 0.3 and ankle_conf > 0.3:
            hip = person_kpts[hip_idx][:2].astype(int)
            knee = person_kpts[knee_idx][:2].astype(int)
            ankle = person_kpts[ankle_idx][:2].astype(int)
            # Draw selected leg
            cv2.line(img_kp, tuple(hip), tuple(knee), (0,255,0), 4)
            cv2.line(img_kp, tuple(knee), tuple(ankle), (0,255,0), 4)
        else:
            print(f"Skipping leg for person due to low confidence: hip={hip_conf:.2f}, knee={knee_conf:.2f}, ankle={ankle_conf:.2f}")
        cv2.circle(img_kp, tuple(hip), 6, (255,0,0), -1)
        cv2.circle(img_kp, tuple(knee), 6, (0,0,255), -1)
        cv2.circle(img_kp, tuple(ankle), 6, (0,255,255), -1)
        out_path = OUT_DIR / f"{img_path.stem}_person{person_id}_leg{leg_idx}_kp.jpg"
        cv2.imwrite(str(out_path), img_kp)
        print(f"Saved: {out_path}")
print("Done.")
