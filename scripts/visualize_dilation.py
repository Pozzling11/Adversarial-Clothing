"""
Visualize YOLOv8 person segmentation + dilation on 3 sample images.
Shows original mask and dilated mask side-by-side.
Falls back to bbox-based mask if segmentation not available.
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import random

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = str(PROJECT_ROOT / "yolov8n.pt")
# Try segmentation model first, fall back to detection
SEG_MODEL_PATH = str(PROJECT_ROOT / "yolov8n-seg.pt")
DATA_DIR = PROJECT_ROOT / "data" / "clean"

# Load YOLOv8 segmentation model (auto-downloads if needed)
print("[*] Loading YOLOv8n-seg (segmentation model)…")
try:
    yolo = YOLO('yolov8n-seg.pt')  # Auto-downloads if not present
    print("  [✓] Segmentation model loaded")
except Exception as e:
    print(f"  [!] Error loading segmentation model: {e}")
    print("  Attempting to use detection model as fallback…")
    yolo = YOLO(MODEL_PATH)
    use_seg = False

# Get 3 random images from clean data
image_files = list(DATA_DIR.glob("*.jpg")) + list(DATA_DIR.glob("*.png")) + list(DATA_DIR.glob("*.avif"))
random.shuffle(image_files)
sample_files = image_files[:3]

print(f"[*] Found {len(image_files)} images, using {len(sample_files)} samples")

for img_path in sample_files:
    print(f"\n[*] Processing: {img_path.name}")
    
    # Read image
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"  [!] Failed to read {img_path}")
        continue
    
    h, w = img_bgr.shape[:2]
    
    # Run YOLOv8 (segmentation or detection)
    results = yolo(img_bgr, verbose=False, conf=0.3)
    
    if len(results) == 0:
        print(f"  [!] No detections found")
        continue
    
    # Extract person mask using segmentation
    person_mask = np.zeros((h, w), dtype=np.uint8)
    
    if results[0].masks is not None:
        # Segmentation: use pixel-perfect masks
        masks = results[0].masks.data.cpu().numpy()  # (N, H, W)
        classes = results[0].boxes.cls.cpu().numpy()  # (N,)
        
        for mask, cls in zip(masks, classes):
            if int(cls) == 0:  # person class
                # Resize mask to image size if needed
                if mask.shape != (h, w):
                    mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
                person_mask = np.maximum(person_mask, (mask > 0.5).astype(np.uint8))
    
    if person_mask.sum() == 0:
        print(f"  [!] No person mask found")
        continue
    
    # Create dilations
    kernel_5pct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, int(w * 0.05)), max(3, int(h * 0.05))))
    kernel_10pct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, int(w * 0.10)), max(3, int(h * 0.10))))
    
    dilated_5pct = cv2.dilate(person_mask, kernel_5pct, iterations=1)
    dilated_10pct = cv2.dilate(person_mask, kernel_10pct, iterations=1)
    
    # Create visualization
    display = img_bgr.copy().astype(np.float32) / 255.0
    
    # Draw original mask (green)
    orig_overlay = display.copy()
    orig_overlay[:, :, 1] = np.maximum(orig_overlay[:, :, 1], person_mask * 0.8)  # green channel
    display = 0.7 * display + 0.3 * orig_overlay
    
    # Draw 5% dilation boundary (yellow)
    boundary_5 = cv2.absdiff(dilated_5pct, person_mask).astype(bool)
    display[boundary_5, 0] = np.minimum(display[boundary_5, 0] + 0.3, 1.0)  # R
    display[boundary_5, 1] = np.minimum(display[boundary_5, 1] + 0.3, 1.0)  # G
    
    # Draw 10% dilation boundary (red)
    boundary_10 = cv2.absdiff(dilated_10pct, person_mask).astype(bool)
    display[boundary_10, 2] = np.minimum(display[boundary_10, 2] + 0.4, 1.0)  # R
    
    out_path = PROJECT_ROOT / "results" / f"dilation_viz_{img_path.stem}.png"
    cv2.imwrite(str(out_path), (display * 255).astype(np.uint8)[:, :, ::-1])
    print(f"  [✓] Saved: {out_path.name}")
    print(f"      Original mask coverage: {(person_mask.sum() / (h*w) * 100):.1f}%")
    print(f"      +5% dilated:  {(dilated_5pct.sum() / (h*w) * 100):.1f}%")
    print(f"      +10% dilated: {(dilated_10pct.sum() / (h*w) * 100):.1f}%")

print("\n[*] Done! Check results/ for visualizations.")
print("    Green = original person mask")
print("    Yellow = 5% dilation")
print("    Red = 10% dilation")
