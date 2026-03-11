import cv2
import numpy as np

IMG_SIZE = 640

# ── Extract mask directly from PotentialShape2.png (A3, 300dpi) ──────────────
raw = cv2.imread("potential shapes/PotentialShape2.png", cv2.IMREAD_GRAYSCALE)
_, shape_bw = cv2.threshold(raw, 128, 255, cv2.THRESH_BINARY_INV)  # dark pixels = shape
coords = np.argwhere(shape_bw > 0)
r0, r1 = int(coords[:,0].min()), int(coords[:,0].max())
c0, c1 = int(coords[:,1].min()), int(coords[:,1].max())
shape_crop = shape_bw[r0:r1+1, c0:c1+1]
print(f"PotentialShape2: {raw.shape[1]}w x {raw.shape[0]}h  |  shape crop: {c1-c0}w x {r1-r0}h")


def detect_person_bbox_simple(img_bgr):
    """Use YOLO to get person bbox. Falls back to a centred 60% bbox."""
    from ultralytics import YOLO
    yolo = YOLO("yolov8n.pt")
    results = yolo.predict(source=img_bgr, conf=0.25, classes=[0], verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        h, w = img_bgr.shape[:2]
        return (int(w*0.2), int(h*0.05), int(w*0.8), int(h*0.95))
    best = int(boxes.conf.cpu().argmax())
    x1, y1, x2, y2 = boxes.xyxy[best].cpu().numpy().astype(int)
    return int(x1), int(y1), int(x2), int(y2)


def detect_chin_row(img_bgr, person_bbox, fallback_frac=0.15):
    px1, py1, px2, py2 = person_bbox
    pbbox_h = py2 - py1
    fallback_row = py1 + int(pbbox_h * fallback_frac)
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=4, minSize=(15, 15)
        )
        face_zone_y2 = py1 + int(pbbox_h * 0.40)
        valid = [
            (fx, fy, fw, fh) for (fx, fy, fw, fh) in faces
            if py1 <= fy + fh // 2 <= face_zone_y2
            and px1 <= fx + fw // 2 <= px2
        ]
        if not valid:
            return fallback_row, 'fallback'
        fx, fy, fw, fh = max(valid, key=lambda f: f[2] * f[3])
        return fy + fh, 'haar'
    except Exception:
        return fallback_row, 'fallback'


def make_preview(img_path, shape_crop, save_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"  SKIP (can't read): {img_path}")
        return
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Detect person bbox
    bbox = detect_person_bbox_simple(img)
    x1, y1, x2, y2 = bbox
    bbox_w = x2 - x1
    cx = (x1 + x2) // 2

    # Scale mask to 85% of bbox width
    v_width  = int(bbox_w * 0.85)
    crop_h, crop_w = shape_crop.shape
    v_height = int(v_width * crop_h / crop_w)

    mask_resized = cv2.resize(shape_crop, (v_width, v_height), interpolation=cv2.INTER_LINEAR)
    _, mask_bin  = cv2.threshold(mask_resized, 64, 255, cv2.THRESH_BINARY)
    mask = (mask_bin / 255.0).astype(np.float32)

    # Chin-anchored placement
    chin_row, chin_method = detect_chin_row(img, bbox)
    centre_col = v_width // 2
    centre_dip_row = 0
    for r in range(v_height):
        if mask[r, centre_col] > 0.5:
            centre_dip_row = r
            break

    row0 = chin_row - centre_dip_row
    col0 = cx - v_width // 2
    # Clamp horizontally only — vertical clamping would displace the arch from the chin
    col0 = max(0, min(col0, IMG_SIZE - v_width))
    # Allow patch to overflow bottom/top; clip the compositing region instead
    img_row0  = max(0, row0)
    img_row1  = min(IMG_SIZE, row0 + v_height)
    mask_row0 = img_row0 - row0   # slice into mask if patch starts above image
    mask_row1 = mask_row0 + (img_row1 - img_row0)

    # Composite
    preview = img.copy()
    roi     = preview[img_row0:img_row1, col0:col0+v_width].astype(np.float32)
    mask_sl = mask[mask_row0:mask_row1, :]
    colour  = np.zeros_like(roi); colour[:] = [0, 0, 220]
    mask3   = np.stack([mask_sl, mask_sl, mask_sl], axis=2)
    blended = roi * (1 - mask3 * 0.6) + colour * mask3 * 0.6
    preview[img_row0:img_row1, col0:col0+v_width] = blended.astype(np.uint8)

    cv2.rectangle(preview, (x1, y1), (x2, y2), (220, 100, 0), 2)
    cv2.circle(preview, (cx, chin_row), 5, (0, 220, 0), -1)
    cv2.putText(preview, f"chin ({chin_method})", (cx+8, chin_row+4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,220,0), 1)

    cv2.imwrite(save_path, preview)
    import os
    print(f"  {os.path.basename(img_path):40s}  chin={chin_method:8s} row={chin_row}  →  {save_path}")


# ── Run on 5 images ───────────────────────────────────────────────────────────
test_images = [
    ("data/clean/woman full body.jpg",       "potential shapes/preview_1_woman_full_body.jpg"),
    ("data/clean/upper body.jpg",            "potential shapes/preview_2_upper_body.jpg"),
    ("data/clean/man very upclose.jpg",      "potential shapes/preview_3_man_upclose.jpg"),
    ("data/clean/blk tee far.JPG",           "potential shapes/preview_4_blk_tee_far.jpg"),
    ("data/clean/full body chill pose.PNG",  "potential shapes/preview_5_full_body_chill.jpg"),
]

print("\nGenerating previews:")
for img_path, save_path in test_images:
    make_preview(img_path, shape_crop, save_path)

print("\nDone.")


# ── Load host image ───────────────────────────────────────────────────────────
host = cv2.imread("data/clean/woman full body.jpg")
host = cv2.resize(host, (IMG_SIZE, IMG_SIZE))

# Simulated realistic full-body person bbox, centred
x1, y1, x2, y2 = 155, 40, 485, 625
bbox_w = x2 - x1
cx = (x1 + x2) // 2

# ── Scale shape to match shoulder width ──────────────────────────────────────
# Shoulders ≈ 75% of the full person bbox width (bbox has ~12% padding each side).
# Keeping the V at or just within shoulder width avoids the over-wide look and
# ensures the outer corners land on the shoulder tips, not past them.
v_width  = int(bbox_w * 0.85)
crop_h, crop_w = shape_crop.shape
v_height = int(v_width * crop_h / crop_w)   # preserve A3 aspect ratio

mask_resized = cv2.resize(shape_crop, (v_width, v_height), interpolation=cv2.INTER_LINEAR)
_, mask_bin  = cv2.threshold(mask_resized, 64, 255, cv2.THRESH_BINARY)
mask = (mask_bin / 255.0).astype(np.float32)
print(f"Mask size: {v_width}w x {v_height}h")

# ── Detect chin: Haar cascade (Option B) with fraction fallback (Option A) ───
def detect_chin_row(img_bgr: np.ndarray, person_bbox: tuple, fallback_frac: float = 0.15) -> tuple[int, str]:
    """
    Returns (chin_row, method) where method is 'haar' or 'fallback'.
    chin_row is the pixel row (in img_bgr coords) of the chin.
    Falls back to fallback_frac * bbox_height below bbox top if face not found.
    """
    px1, py1, px2, py2 = person_bbox
    pbbox_h = py2 - py1
    fallback_row = py1 + int(pbbox_h * fallback_frac)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    # Search full image — then filter for faces whose centre falls within the
    # upper 40% of the person bbox (face zone).
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=4, minSize=(15, 15)
    )
    px1, py1, px2, py2 = person_bbox
    pbbox_h = py2 - py1
    face_zone_y2 = py1 + int(pbbox_h * 0.40)
    valid = []
    for (fx, fy, fw, fh) in faces:
        face_cy = fy + fh // 2
        if py1 <= face_cy <= face_zone_y2 and px1 <= fx + fw // 2 <= px2:
            valid.append((fx, fy, fw, fh))
    if not valid:
        return fallback_row, 'fallback'
    largest = max(valid, key=lambda f: f[2] * f[3])
    fx, fy, fw, fh = largest
    chin_row = fy + fh   # bottom of face bbox = chin
    return chin_row, 'haar'

person_bbox = (x1, y1, x2, y2)
chin_row, chin_method = detect_chin_row(host, person_bbox)
print(f"Chin detected via    : {chin_method}  →  row {chin_row}")

# ── Align centre-dip of top curve to the detected chin ───────────────────────
# Find the row in the mask where the centre column first turns on (top of curve dip)
centre_col = v_width // 2
centre_dip_row = 0
for r in range(v_height):
    if mask[r, centre_col] > 0.5:
        centre_dip_row = r
        break

row0 = chin_row - centre_dip_row     # shift so centre-dip aligns with chin
col0 = cx - v_width // 2
row0 = max(0, min(row0, IMG_SIZE - v_height))
col0 = max(0, min(col0, IMG_SIZE - v_width))
print(f"Centre dip in mask   : row {centre_dip_row}  →  placed at row0={row0}, col0={col0}")

# ── Composite onto host as semi-transparent red overlay ───────────────────────
preview = host.copy()
roi     = preview[row0:row0+v_height, col0:col0+v_width].astype(np.float32)
colour  = np.zeros_like(roi)
colour[:] = [0, 0, 220]   # red in BGR
mask3   = np.stack([mask, mask, mask], axis=2)
blended = roi * (1 - mask3 * 0.6) + colour * mask3 * 0.6
preview[row0:row0+v_height, col0:col0+v_width] = blended.astype(np.uint8)

# Person bbox in blue
cv2.rectangle(preview, (x1, y1), (x2, y2), (220, 100, 0), 2)
cv2.putText(preview, "person bbox",   (x1+4, y1-6),               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,100,0), 1)
cv2.putText(preview, "V-shape patch", (col0+4, row0+v_height+18),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,220), 1)
# Mark chin point (green dot + label)
cv2.circle(preview, (cx, chin_row), 5, (0, 220, 0), -1)
cv2.putText(preview, f"chin ({chin_method})", (cx+8, chin_row+4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,220,0), 1)

cv2.imwrite("potential shapes/v_shape_preview.jpg", preview)
cv2.imwrite("potential shapes/v_shape_mask.png", (mask * 255).astype(np.uint8))
print(f"Saved: potential shapes/v_shape_preview.jpg  (placed at row={row0}, col={col0})")
print("Saved: potential shapes/v_shape_mask.png")

