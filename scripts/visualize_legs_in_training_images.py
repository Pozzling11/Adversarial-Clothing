import os
import cv2
import numpy as np
from ultralytics import YOLO

# Directory paths
input_dir = 'data/TRAINING LEG IMAGES'
output_dir = 'leg_pose_vis'
os.makedirs(output_dir, exist_ok=True)

# Load YOLOv8 pose model
model = YOLO('yolov8n-pose.pt')

# Keypoint indices for leg (user provided)
LEG_INDICES = [12, 14, 16]  # HIP, KNEE, ANKLE

# Confidence threshold
CONF_THRESH = 0.3

# Process each image in the input directory
for fname in os.listdir(input_dir):
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.avif')):
        continue
    img_path = os.path.join(input_dir, fname)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load {img_path}")
        continue
    results = model(img)
    keypoints = results[0].keypoints.data.cpu().numpy() if results[0].keypoints is not None else []
    img_vis = img.copy()
    for person_kpts in keypoints:
        # Draw all keypoints
        for idx, (x, y, conf) in enumerate(person_kpts):
            pt = (int(x), int(y))
            cv2.circle(img_vis, pt, 4, (0, 0, 255), -1)
            cv2.putText(img_vis, str(idx), pt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
        # Draw leg if all keypoints are present with sufficient confidence
        hip_idx, knee_idx, ankle_idx = LEG_INDICES
        hip_conf = person_kpts[hip_idx][2]
        knee_conf = person_kpts[knee_idx][2]
        ankle_conf = person_kpts[ankle_idx][2]
        if hip_conf > CONF_THRESH and knee_conf > CONF_THRESH and ankle_conf > CONF_THRESH:
            hip = person_kpts[hip_idx][:2].astype(float)
            knee = person_kpts[knee_idx][:2].astype(float)
            ankle = person_kpts[ankle_idx][:2].astype(float)
            # Rectangle parameters
            leg_length = np.linalg.norm(ankle - hip)
            rect_width = leg_length * 0.15 * 1.3  # Increase width by 1.3x
            rect_height = leg_length       # Height = distance hip-ankle
            # Center and angle
            center = (hip + ankle) / 2
            angle = np.degrees(np.arctan2(ankle[1] - hip[1], ankle[0] - hip[0]))
            # Get rectangle points
            box = cv2.boxPoints(((center[0], center[1]), (rect_height, rect_width), angle))
            box = box.astype(np.int32)
            cv2.drawContours(img_vis, [box], 0, (255, 0, 255), 2)
            # Draw lines
            cv2.line(img_vis, tuple(hip.astype(int)), tuple(knee.astype(int)), (0,255,0), 3)
            cv2.line(img_vis, tuple(knee.astype(int)), tuple(ankle.astype(int)), (0,255,0), 3)
        else:
            print(f"{fname}: Skipping leg for person due to low confidence: hip={hip_conf:.2f}, knee={knee_conf:.2f}, ankle={ankle_conf:.2f}")
    out_path = os.path.join(output_dir, fname)
    cv2.imwrite(out_path, img_vis)
    print(f"Saved visualization to {out_path}")
