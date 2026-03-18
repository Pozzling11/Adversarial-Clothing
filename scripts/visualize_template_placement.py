
import cv2
import numpy as np
from pathlib import Path
from scripts.template_crop_utils import crop_template_to_visible_region

# --- YOLOv8 segmentation ---
from ultralytics import YOLO

IMG_DIR = Path('data/clean')
TEMPLATE_PATH = 'data/human body chin down silhouette.png'  # Update if needed
OUT_DIR = Path('template_placement_vis')
OUT_DIR.mkdir(exist_ok=True)

# Load template
template = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
assert template is not None, f"Template not found: {TEMPLATE_PATH}"

# Load YOLOv8n-seg model
yolo = YOLO('yolov8n-seg.pt')

# Get 10 images
img_files = [f for f in IMG_DIR.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']]
img_files = img_files[:10]


for img_path in img_files:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Failed to load {img_path}")
        continue
    h, w = img.shape[:2]
    # Run YOLOv8 segmentation
    results = yolo(img, verbose=False)
    seg_mask = np.zeros((h, w), dtype=np.uint8)
    if results and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()  # (N, H, W)
        classes = results[0].boxes.cls.cpu().numpy()  # (N,)
        for mask, cls in zip(masks, classes):
            if int(cls) == 0:  # person class
                if mask.shape != (h, w):
                    mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
                seg_mask = np.maximum(seg_mask, (mask > 0.5).astype(np.uint8))
    if seg_mask.sum() == 0:
        print(f"No person mask found for {img_path.name}, skipping.")
        continue

    # --- Debug: Save raw segmentation mask ---
    cv2.imwrite(str(OUT_DIR / f"{img_path.stem}_seg_mask.png"), seg_mask * 255)
    cv2.imwrite(str(OUT_DIR / f"{img_path.stem}_raw_template.png"), template)

    # --- Find horizontal center of person in mask ---
    col_sum = seg_mask.sum(axis=0)
    center_col = int(np.round(np.argmax(col_sum)))
    # Find chin row: first row with body pixel at center_col
    chin_row = np.argmax(seg_mask[:, center_col] > 0)

    # --- Find center top of template ---
    template_center_col = template.shape[1] // 2
    template_chin_row = np.argmax(template[:, template_center_col] < 128)
    # Crop template from chin down
    template_body = template[template_chin_row:, :]
    template_body_h, template_body_w = template_body.shape

    # --- Compute visible body height in mask (from chin_row to lowest body pixel at center_col) ---
    mask_rows = np.where(seg_mask[:, center_col] > 0)[0]
    if len(mask_rows) == 0:
        print(f"No body pixels at center_col for {img_path.name}, skipping.")
        continue
    mask_bottom = mask_rows[-1]
    visible_body_h = mask_bottom - chin_row + 1
    # --- Scale template body to match visible body height, preserve aspect ratio ---
    scale = visible_body_h / template_body_h
    new_w = int(template_body_w * scale)
    new_h = visible_body_h
    template_scaled = cv2.resize(template_body, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # --- Overlay: anchor template center top to (chin_row, center_col) ---
    overlay = img.copy()
    x0 = center_col - new_w // 2
    x1 = x0 + new_w
    y0 = chin_row
    y1 = y0 + new_h
    # Clip to image bounds
    x0_clip, x1_clip = max(0, x0), min(w, x1)
    y0_clip, y1_clip = max(0, y0), min(h, y1)
    # Compute region in template_scaled
    tx0 = x0_clip - x0
    tx1 = tx0 + (x1_clip - x0_clip)
    ty0 = y0_clip - y0
    ty1 = ty0 + (y1_clip - y0_clip)
    mask = template_scaled[ty0:ty1, tx0:tx1] < 128
    overlay[y0_clip:y1_clip, x0_clip:x1_clip, 2][mask] = 255  # Red channel
    vis = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
    out_path = OUT_DIR / f"{img_path.stem}_template_vis.jpg"
    cv2.imwrite(str(out_path), vis)
    print(f"Saved: {out_path}")
print("Done.")
