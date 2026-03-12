"""
generate_patch.py  (v2 — EOT-in-training / multi-host / random torso placement)
================================================================================
Gradient-based adversarial patch synthesis targeting the 'person' class in YOLOv8n.

v2 upgrades over v1
-------------------
  1. --patch-size  : configurable patch size (default 256px for stronger signal)
  2. --hosts-dir   : pool of background images; a random host is sampled each step
                     so the patch generalises across backgrounds / clothing / angles
  3. EOT in training: each forward pass applies random scale, rotation, brightness
                     and blur to the *composite* before feeding to the model —
                     pushing the patch to survive real-world variation
  4. Random torso placement per step: row/col are jittered within the torso band
                     every iteration to prevent the patch overfitting one location

Strategy: PGD (Projected Gradient Descent) over the patch pixels.
  Loss = mean(top-k person confidences) + max(top-k person confidences)
  Minimising drives the model toward a False Negative on every augmented view.

Ultralytics 8.x raw output: (batch, 4+nc, 8400)  channels-first.
  channel 4 = person confidence for all 8400 anchors.

Usage
-----
  python scripts/generate_patch.py
  python scripts/generate_patch.py --patch-size 256 --steps 1500 --lr 0.02 --verbose
  python scripts/generate_patch.py --hosts-dir data/clean --steps 1000
"""

import argparse
import random
import sys
from pathlib import Path

import math

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT           = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS        = str(PROJECT_ROOT / "yolov8n.pt")
DEFAULT_HOSTS          = str(PROJECT_ROOT / "data" / "clean")
DEFAULT_PRINTABLE_COLS = str(PROJECT_ROOT / "data" / "printable_colors.txt")
PERSON_CLASS    = 0
PERSON_COL_IDX  = 4 + PERSON_CLASS   # channel index in (4+nc, N_anchors) layout
TOP_K           = 50
IMG_SIZE        = 640
SUPPORTED_EXTS  = {".jpg", ".jpeg", ".png", ".bmp"}

# EOT hyper-parameters applied to the composite during training
EOT_SCALE_RANGE  = (0.35, 1.30)   # zoom range — simulates different viewing distances (0.35 = ~3× farther away)
EOT_ROT_RANGE    = 20.0           # degrees — wider rotation for head/body tilt
EOT_BLUR_MAX     = 5              # max Gaussian kernel size
EOT_BRIGHTNESS   = 0.50          # ± fraction — wide swing covers deep shade → bright sun
EOT_PERSP_JITTER = 0.35          # max fractional corner displacement for perspective warp
EOT_JPEG_QUALITY = (40, 95)      # random JPEG quality range (simulates camera compression)
EOT_COLOR_JITTER = 0.20          # per-channel ± brightness shift (simulates ink colour shift)
EOT_PRINT_NOISE  = 10.0          # max std-dev of Gaussian noise added (simulates print grain)
EOT_HSV_HUE      = 25            # ± hue shift in degrees (covers sodium / daylight / fluorescent)
EOT_HSV_SAT      = 0.45          # ± saturation multiplier (washed-out overexposure → vivid)
EOT_SHADOW_PROB  = 0.65          # probability of adding a random shadow strip
EOT_GAMMA_RANGE  = (0.45, 2.2)   # gamma ∈ (0,1) brightens (overexposed); >1 darkens (underexposed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adversarial patch generator (PGD v2) for YOLOv8n")
    p.add_argument("--model",      default=DEFAULT_WEIGHTS,
                   help="Path to YOLOv8n .pt weights")
    p.add_argument("--hosts-dir",  default=DEFAULT_HOSTS,
                   help="Dir of background images; one is sampled randomly each step")
    p.add_argument("--host",       default=None,
                   help="Single background image (overrides --hosts-dir)")
    p.add_argument("--out",        default=None,
                   help="Output PNG path; defaults to patterns/patch_<size>.png")
    p.add_argument("--patch-size", type=int, default=256,
                   help="Patch side length in pixels (default 256)")
    p.add_argument("--steps",      type=int, default=1500,
                   help="PGD optimisation steps (default 1500)")
    p.add_argument("--lr",         type=float, default=0.02,
                   help="PGD step size (default 0.02)")
    p.add_argument("--eps",        type=float, default=1.0,
                   help="L-inf budget per pixel in [0,1] (default 1.0 = unconstrained)")
    p.add_argument("--topk",       type=int, default=TOP_K,
                   help="Top-k anchors used in loss (default 50)")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Images averaged per PGD step for smoother gradients (default 4)")
    p.add_argument("--no-eot",     action="store_true",
                   help="Disable EOT augmentation during training")
    p.add_argument("--alpha",      type=float, default=0.01,
                   help="Weight for NPS loss  (default 0.01; 0 = disabled)")
    p.add_argument("--beta",       type=float, default=2.5,
                   help="Weight for TV  loss  (default 2.5;  0 = disabled)")
    p.add_argument("--printable-colors", default=DEFAULT_PRINTABLE_COLS,
                   help="Path to printable colours file (R G B floats, one per line)")
    p.add_argument("--init",       default="uniform",
                   choices=["uniform", "gaussian", "checkerboard", "stripes",
                            "salt_pepper", "gray", "blocky", "perlin"],
                   help="Patch initialisation pattern (default: uniform)")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--resume",     action="store_true",
                   help="Resume from checkpoint if one exists for this patch name")
    p.add_argument("--checkpoint-every", type=int, default=100,
                   help="Save a resume checkpoint every N steps (default 100)")
    p.add_argument("--verbose",    action="store_true",
                   help="Print loss every 50 steps")
    p.add_argument("--target-image", default=None,
                   help="Path to target image; patch is guided toward this aesthetic")
    p.add_argument("--style-weight", type=float, default=0.5,
                   help="Weight for content/style loss (default 0.5; 0 = disabled)")
    p.add_argument("--bbox-placement", action="store_true",
                   help="Detect person bbox each step and place+scale patch relative to it."
                        " Prevents the patch training on occlusion rather than adversarial signal.")
    p.add_argument("--patch-fraction", type=float, default=0.25,
                   help="When --bbox-placement is set: patch composite height as a fraction of"
                        " the person bbox height (default 0.25 ≈ 25%% of person height,"
                        " realistic t-shirt chest print size).")
    p.add_argument("--hat-patch", action="store_true",
                   help="Also train a second adversarial crown patch placed on top of the head."
                        " Strictly clamped to the crown zone (top hat-fraction of bbox) so it"
                        " NEVER covers the face — avoids occlusion cheating.")
    p.add_argument("--hat-fraction", type=float, default=0.08,
                   help="Hat patch height as a fraction of person bbox height (default 0.08)."
                        " Hard-clamped at ≤0.12 to stay above the face region.")
    p.add_argument("--iou-loss", action="store_true",
                   help="Weight each anchor's person score by a Gaussian centred on the detected"
                        " person bbox before top-k selection.  Focuses the adversarial gradient on"
                        " anchors that genuinely overlap the person rather than background noise."
                        " Recommended with --bbox-placement for cleanest signal.")
    p.add_argument("--iou-sigma", type=float, default=0.5,
                   help="Gaussian sigma for IoU weighting, in units of bbox half-size (default 0.5)."
                        " Smaller = tighter focus on bbox centre; larger = broader coverage.")
    p.add_argument("--hard-mining", action="store_true",
                   help="Hard-example mining: sample host images with probability proportional"
                        " to their recent per-image loss EMA.  Focuses training on frames the"
                        " current patch fails to fool — analogous to sequence-level curriculum"
                        " learning from the 2511.16020 paper without requiring video sequences.")
    p.add_argument("--hard-temp", type=float, default=0.5,
                   help="Softmax temperature for hard-mining host sampling (default 0.5)."
                        " Lower = more concentrated on hardest hosts; higher = more uniform.")
    p.add_argument("--letter", default=None,
                   help="Embed a visible character in the patch (e.g. 'A'). The patch luminance"
                        " is softly guided to form this letterform while adversarial signal"
                        " remains dominant. No colour constraints — only luminance contrast.")
    p.add_argument("--letter-weight", type=float, default=0.2,
                   help="Weight for letter shape loss (default 0.2). Increase toward 0.5 for"
                        " a clearer letter at the cost of slightly weaker adversarial effect.")
    # V-shape masked patch
    p.add_argument("--v-shape", action="store_true",
                   help="Use a V-shaped (shoulder/torso) patch mask instead of a square patch."
                        " Requires --bbox-placement. Mask is loaded from --v-shape-mask.")
    p.add_argument("--v-shape-mask", default="potential shapes/PotentialShape2.png",
                   help="Path to V-shape reference PNG (dark shape on white background).")
    p.add_argument("--v-width-frac", type=float, default=0.85,
                   help="V-shape width as a fraction of the person bbox width (default 0.85).")
    p.add_argument("--chin-fallback-frac", type=float, default=0.15,
                   help="Fallback chin row as a fraction of bbox height below the top of the"
                        " bbox, used when Haar face detection fails (default 0.15).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Content / style loss
# ---------------------------------------------------------------------------

def content_loss(patch: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    L2 pixel loss between the current patch and a target style image.
    Both are (1,3,H,W) in [0,1]. Drives the patch appearance toward the target
    while the adversarial loss drives its function.
    """
    return torch.nn.functional.mse_loss(patch, target)


# ---------------------------------------------------------------------------
# Image / tensor helpers
# ---------------------------------------------------------------------------

def load_bgr(path: str, size: int = IMG_SIZE) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"[ERROR] Cannot read image: {path}")
    return cv2.resize(img, (size, size))


def preprocess(bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 → float32 (1,3,H,W) [0,1]."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)


def load_host_pool(hosts_dir: str, single_host: str | None) -> list[np.ndarray]:
    """Return a list of preprocessed host BGR images at IMG_SIZE resolution."""
    if single_host:
        return [load_bgr(single_host)]
    d = Path(hosts_dir)
    if not d.exists():
        print(f"[WARN] --hosts-dir not found: {d}. Using blank canvas.")
        return [np.full((IMG_SIZE, IMG_SIZE, 3), 114, dtype=np.uint8)]
    imgs = [
        load_bgr(str(p))
        for p in sorted(d.iterdir())
        if p.suffix.lower() in SUPPORTED_EXTS and not p.name.startswith("._")
    ]
    if not imgs:
        print("[WARN] No images found in hosts-dir. Using blank canvas.")
        return [np.full((IMG_SIZE, IMG_SIZE, 3), 114, dtype=np.uint8)]
    print(f"[INFO] Host pool  : {len(imgs)} image(s) from {d}")
    return imgs


def apply_patch_to_tensor(
    img_t: torch.Tensor,    # (1,3,H,W) float32 [0,1]
    patch_t: torch.Tensor,  # (1,3,ph,pw) float32 [0,1]  — must keep grad
    row: int,
    col: int,
) -> torch.Tensor:
    ph, pw = patch_t.shape[2], patch_t.shape[3]
    # Use index_put for in-place safety with autograd
    out = img_t.clone()
    out[:, :, row:row + ph, col:col + pw] = patch_t
    return out


def apply_patch_resized(
    img_t: torch.Tensor,    # (1,3,H,W) float32 [0,1]
    patch_t: torch.Tensor,  # (1,3,ph,pw) float32 [0,1]  — must keep grad
    row: int,
    col: int,
    composite_size: int,    # resize patch to this square before compositing
) -> torch.Tensor:
    """
    Differentiably resize patch_t to composite_size×composite_size, then
    composite it onto img_t at (row, col).  Gradients flow back through
    F.interpolate so the optimiser still updates the canonical patch.
    """
    if composite_size != patch_t.shape[2] or composite_size != patch_t.shape[3]:
        patch_scaled = torch.nn.functional.interpolate(
            patch_t,
            size=(composite_size, composite_size),
            mode="bilinear",
            align_corners=False,
        )
    else:
        patch_scaled = patch_t
    out = img_t.clone()
    out[:, :, row:row + composite_size, col:col + composite_size] = patch_scaled
    return out


def detect_person_bbox(
    yolo_wrapper,
    host_bgr: np.ndarray,
    conf_thresh: float = 0.25,
) -> tuple[int, int, int, int] | None:
    """
    Run a no-grad YOLOv8 inference pass on host_bgr and return the
    highest-confidence person bbox as (x1, y1, x2, y2) pixel coords,
    or None if no person is found above conf_thresh.
    """
    with torch.no_grad():
        results = yolo_wrapper.predict(
            source=host_bgr,
            conf=conf_thresh,
            classes=[PERSON_CLASS],
            verbose=False,
        )
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None
    best = int(boxes.conf.cpu().argmax())
    x1, y1, x2, y2 = boxes.xyxy[best].cpu().numpy().astype(int)
    return int(x1), int(y1), int(x2), int(y2)


def detect_chin_row(
    img_bgr: np.ndarray,
    person_bbox: tuple[int, int, int, int],
    fallback_frac: float = 0.15,
) -> tuple[int, str]:
    """
    Detect the chin row for a person in img_bgr.

    Strategy (Option B → Option A fallback):
      1. Run OpenCV Haar cascade face detection on the full image.
      2. Keep only faces whose centre falls within the upper 40% of the person
         bbox — eliminates false positives from other people in the frame.
      3. Chin = bottom edge of the largest qualifying face rectangle.
      4. If no face is found, fall back to fallback_frac × bbox_height below
         the top of the person bbox (Option A, default 15%).

    Returns
    -------
    (chin_row, method) where method is 'haar' or 'fallback'.
    chin_row is the absolute pixel row in img_bgr coordinates.
    """
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


# ---------------------------------------------------------------------------
# V-shape mask helpers
# ---------------------------------------------------------------------------

def load_v_shape_mask(path: str) -> tuple[np.ndarray, int]:
    """
    Load a V-shape mask from a PNG file (dark shape on white background).

    Returns
    -------
    (shape_crop, centre_dip_row)
        shape_crop     : (H, W) uint8 array, 255 inside V and 0 outside,
                         tightly cropped to the shape bounding box.
        centre_dip_row : first row in shape_crop where the centre column
                         is active — used to anchor the arch under the chin.
    """
    raw = cv2.imread(str(PROJECT_ROOT / path) if not Path(path).is_absolute() else path,
                     cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(f"V-shape mask PNG not found: {path}")
    _, bw = cv2.threshold(raw, 128, 255, cv2.THRESH_BINARY_INV)
    coords = np.argwhere(bw > 0)
    if coords.size == 0:
        raise ValueError(f"V-shape mask PNG contains no dark pixels: {path}")
    r0, r1 = int(coords[:, 0].min()), int(coords[:, 0].max())
    c0, c1 = int(coords[:, 1].min()), int(coords[:, 1].max())
    shape_crop = bw[r0:r1 + 1, c0:c1 + 1]
    crop_h, crop_w = shape_crop.shape
    centre_col = crop_w // 2
    centre_dip_row = 0
    for r in range(crop_h):
        if shape_crop[r, centre_col] > 128:
            centre_dip_row = r
            break
    return shape_crop, centre_dip_row


def v_shape_placement(
    bbox: tuple[int, int, int, int],
    chin_row: int,
    v_width_frac: float,
    img_size: int,
    shape_crop: np.ndarray,
    centre_dip_row: int,
) -> tuple[int, int, int, int, np.ndarray]:
    """
    Compute chin-anchored V-shape placement for a person bbox.

    The arch of the V is pinned to chin_row; the shape is scaled to
    v_width_frac × bbox_width preserving the aspect ratio of shape_crop.

    Returns
    -------
    (row0, col0, v_width, v_height, mask_bin)
        row0/col0  : top-left of the bounding rectangle in image coords
                     (row0 may be negative — caller must clip when compositing).
        v_width/v_height : pixel dimensions of the scaled mask.
        mask_bin   : (v_height, v_width) uint8 array, 255 inside V, 0 outside.
    """
    x1, y1, x2, y2 = bbox
    bbox_w = max(x2 - x1, 1)
    cx = (x1 + x2) // 2
    crop_h, crop_w = shape_crop.shape
    v_width  = max(16, int(bbox_w * v_width_frac))
    v_height = max(16, int(v_width * crop_h / crop_w))
    mask_r   = cv2.resize(shape_crop, (v_width, v_height), interpolation=cv2.INTER_LINEAR)
    _, mask_bin = cv2.threshold(mask_r, 64, 255, cv2.THRESH_BINARY)
    scaled_dip = int(centre_dip_row * v_height / crop_h)
    row0 = chin_row - scaled_dip
    col0 = max(0, min(cx - v_width // 2, img_size - v_width))
    return row0, col0, v_width, v_height, mask_bin


def apply_v_shape_patch(
    host_t: torch.Tensor,
    patch_t: torch.Tensor,
    row0: int,
    col0: int,
    v_width: int,
    v_height: int,
    mask_f: torch.Tensor,       # (1,1,v_height,v_width) float32 on device — pre-built
    device: torch.device,
) -> torch.Tensor:
    """
    Differentiably composite a V-shaped patch onto a host image tensor.

    patch_t is resized to (v_height, v_width) via F.interpolate so gradients
    flow back to the canonical patch.  mask_f gates which pixels are replaced.
    The ROI is clipped to the image boundary so partial overflow is handled.
    """
    import torch.nn.functional as F
    patch_scaled = F.interpolate(
        patch_t, size=(v_height, v_width), mode="bilinear", align_corners=False
    )  # (1, 3, v_height, v_width)

    img_h = host_t.shape[2]
    img_r0 = max(0, row0)
    img_r1 = min(img_h, row0 + v_height)
    msk_r0 = img_r0 - row0
    msk_r1 = msk_r0 + (img_r1 - img_r0)
    if img_r1 <= img_r0:
        return host_t  # patch entirely above/below image

    out = host_t.clone()
    roi   = out[:, :, img_r0:img_r1, col0:col0 + v_width]
    p_sl  = patch_scaled[:, :, msk_r0:msk_r1, :]
    m_sl  = mask_f[:, :, msk_r0:msk_r1, :]
    out[:, :, img_r0:img_r1, col0:col0 + v_width] = roi * (1.0 - m_sl) + p_sl * m_sl
    return out


def _apply_v_shape_bgr(
    img_bgr: np.ndarray,
    patch_bgr_full: np.ndarray,
    row0: int,
    col0: int,
    v_width: int,
    v_height: int,
    mask_bin: np.ndarray,
) -> np.ndarray:
    """
    Numpy equivalent of apply_v_shape_patch for use in eval / visualisation
    functions that work with BGR arrays rather than tensors.
    """
    p_bgr = cv2.resize(patch_bgr_full, (v_width, v_height), interpolation=cv2.INTER_LINEAR)
    mask_f = (mask_bin.astype(np.float32) / 255.0)[:, :, np.newaxis]  # (H, W, 1)
    img_h = img_bgr.shape[0]
    img_r0 = max(0, row0)
    img_r1 = min(img_h, row0 + v_height)
    msk_r0 = img_r0 - row0
    msk_r1 = msk_r0 + (img_r1 - img_r0)
    if img_r1 <= img_r0:
        return img_bgr
    out = img_bgr.copy()
    roi   = out[img_r0:img_r1, col0:col0 + v_width].astype(np.float32)
    p_sl  = p_bgr[msk_r0:msk_r1, :].astype(np.float32)
    m_sl  = mask_f[msk_r0:msk_r1, :]
    out[img_r0:img_r1, col0:col0 + v_width] = (
        roi * (1.0 - m_sl) + p_sl * m_sl
    ).clip(0, 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Anchor grid + IoU-guided weighting + hard-mining helpers
# ---------------------------------------------------------------------------

def generate_anchor_centers(img_size: int = IMG_SIZE, device: torch.device = None) -> torch.Tensor:
    """
    Generate the 8400 anchor cell centres used by YOLOv8 at img_size×img_size.

    YOLOv8 is anchor-free but uses a dense grid of 8400 cell centres across
    three detection heads at strides 8, 16, 32:
        stride  8 → 80×80  = 6400 cells
        stride 16 → 40×40  = 1600 cells
        stride 32 → 20×20  =  400 cells

    Returns (8400, 2) float32 tensor of (x, y) pixel coords on `device`.
    """
    centres = []
    for stride in [8, 16, 32]:
        n = img_size // stride
        for r in range(n):
            for c in range(n):
                centres.append([(c + 0.5) * stride, (r + 0.5) * stride])
    t = torch.tensor(centres, dtype=torch.float32)
    return t.to(device) if device is not None else t


def bbox_iou_weights(
    bbox: tuple[int, int, int, int],
    anchor_centers: torch.Tensor,  # (N, 2)
    sigma: float = 0.5,
) -> torch.Tensor:
    """
    Compute a soft Gaussian weight for each anchor based on how close its
    centre is to the detected person bbox centre.

    Distance is normalised by the bbox half-dimensions so `sigma=0.5` means
    an anchor at the bbox edge gets weight ≈ exp(-0.5) ≈ 0.6, while an anchor
    one full bbox-width away gets weight ≈ exp(-2) ≈ 0.14.

    Returns (N,) float32 weights ∈ (0, 1] — gradients flow through these
    multiplied scores, focusing the optimiser on person-overlapping anchors.
    """
    x1, y1, x2, y2 = bbox
    bw   = max(x2 - x1, 1)
    bh   = max(y2 - y1, 1)
    bcx  = (x1 + x2) / 2.0
    bcy  = (y1 + y2) / 2.0
    dx   = (anchor_centers[:, 0] - bcx) / bw
    dy   = (anchor_centers[:, 1] - bcy) / bh
    return torch.exp(-(dx ** 2 + dy ** 2) / (2.0 * sigma ** 2))


def _softmax_weights(arr: np.ndarray, temp: float) -> list:
    """
    Compute softmax(arr / temp) and return as a Python list suitable for
    random.choices weights.  Numerically stable via max-subtraction.
    """
    x = arr / max(temp, 1e-8)
    x = x - x.max()
    e = np.exp(x)
    return (e / e.sum()).tolist()


# ---------------------------------------------------------------------------
# EOT transforms  (applied to composite TENSOR — numpy ops via detach/reattach)
# ---------------------------------------------------------------------------

def eot_augment_tensor(composite: torch.Tensor) -> torch.Tensor:
    """
    Apply random photometric + geometric EOT transforms to the composite tensor.
    Gradients DO flow through this because we reattach after numpy ops using
    requires_grad forwarding — the patch pixels are already embedded, so the
    photometric jitter still differentiates through them correctly.

    NOTE: geometric ops (rotation/scale) are applied with INTER_LINEAR, which
    is differentiable in the limit; here we treat them as fixed augmentations
    per step (stop-gradient on transforms, gradient w.r.t. patch pixels only).
    """
    # Convert to numpy for OpenCV ops
    needs_grad = composite.requires_grad
    arr = composite.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()  # HWC [0,1]
    arr = (arr * 255).astype(np.float32)

    h, w = arr.shape[:2]

    # 1. Random brightness / contrast jitter — wide swing for indoor/outdoor
    alpha = 1.0 + random.uniform(-EOT_BRIGHTNESS, EOT_BRIGHTNESS)
    beta  = random.uniform(-40, 40)
    arr = np.clip(arr * alpha + beta, 0, 255)

    # 1b. Gamma correction — simulates over/under exposure (gamma<1=bright, gamma>1=dark)
    gamma = random.uniform(*EOT_GAMMA_RANGE)
    arr = np.clip(np.power(arr / 255.0, gamma) * 255.0, 0, 255)

    # 2. HSV hue + saturation jitter — simulates warm/cool/fluorescent lighting
    arr_uint8 = np.clip(arr, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(arr_uint8, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-EOT_HSV_HUE, EOT_HSV_HUE)) % 180
    sat_scale = 1.0 + random.uniform(-EOT_HSV_SAT, EOT_HSV_SAT)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0, 255)
    arr = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    # 3. Random Gaussian blur — simulates camera out-of-focus / motion
    k = random.choice([1, 1, 3, 3, EOT_BLUR_MAX if EOT_BLUR_MAX % 2 == 1 else EOT_BLUR_MAX + 1])
    if k > 1:
        arr = cv2.GaussianBlur(arr, (k, k), 0)

    # 4. Random scale (zoom) — simulates different viewing distances
    sc = random.uniform(*EOT_SCALE_RANGE)
    new_w, new_h = max(1, int(w * sc)), max(1, int(h * sc))
    scaled = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if sc >= 1.0:
        # Zoomed in — crop centre back to original size
        x0 = (new_w - w) // 2
        y0 = (new_h - h) // 2
        arr = scaled[y0:y0 + h, x0:x0 + w]
    else:
        # Zoomed out — pad with replicated border
        pad_x = (w - new_w) // 2
        pad_y = (h - new_h) // 2
        arr = cv2.copyMakeBorder(scaled, pad_y, h - new_h - pad_y,
                                  pad_x, w - new_w - pad_x, cv2.BORDER_REPLICATE)

    # 5. Random rotation — wider range for body/head tilt
    angle = random.uniform(-EOT_ROT_RANGE, EOT_ROT_RANGE)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    arr = cv2.warpAffine(arr, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)

    # 6. Random perspective warp — simulates patch viewed at an angle
    jx = EOT_PERSP_JITTER * w
    jy = EOT_PERSP_JITTER * h
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_pts = np.float32([
        [random.uniform(0, jx),     random.uniform(0, jy)],
        [random.uniform(w - jx, w), random.uniform(0, jy)],
        [random.uniform(w - jx, w), random.uniform(h - jy, h)],
        [random.uniform(0, jx),     random.uniform(h - jy, h)],
    ])
    P = cv2.getPerspectiveTransform(src_pts, dst_pts)
    arr = cv2.warpPerspective(arr, P, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)

    # 7. Random shadow strip — simulates patch partially in shade
    if random.random() < EOT_SHADOW_PROB:
        shadow_intensity = random.uniform(0.3, 0.7)
        if random.random() < 0.5:  # horizontal strip
            y1 = random.randint(0, h // 2)
            y2 = random.randint(h // 2, h)
            arr[y1:y2, :] = arr[y1:y2, :] * shadow_intensity
        else:  # vertical strip
            x1 = random.randint(0, w // 2)
            x2 = random.randint(w // 2, w)
            arr[:, x1:x2] = arr[:, x1:x2] * shadow_intensity

    # 8. JPEG compression simulation
    quality = random.randint(EOT_JPEG_QUALITY[0], EOT_JPEG_QUALITY[1])
    arr_uint8 = np.clip(arr, 0, 255).astype(np.uint8)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode(".jpg", arr_uint8, encode_param)
    arr = cv2.imdecode(enc, cv2.IMREAD_COLOR).astype(np.float32)

    # 9. Per-channel colour jitter — simulates ink/print colour shift
    for c in range(3):
        shift = random.uniform(-EOT_COLOR_JITTER * 255, EOT_COLOR_JITTER * 255)
        arr[:, :, c] = np.clip(arr[:, :, c] + shift, 0, 255)

    # 10. Gaussian print grain noise
    if EOT_PRINT_NOISE > 0:
        noise_std = random.uniform(0, EOT_PRINT_NOISE)
        arr = np.clip(arr + np.random.normal(0, noise_std, arr.shape).astype(np.float32), 0, 255)

    # Back to tensor
    arr = arr / 255.0
    t = torch.from_numpy(arr.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    t = t.to(composite.device)
    return t


# ── Helpers for differentiable EOT ───────────────────────────────────────────

def _gaussian_kernel_2d(k: int, device: torch.device) -> torch.Tensor:
    """Return a (3, 1, k, k) Gaussian kernel for depthwise conv2d."""
    sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8           # match OpenCV default
    x = torch.arange(k, dtype=torch.float32, device=device) - k // 2
    g = torch.exp(-x ** 2 / (2.0 * sigma ** 2))
    g = g / g.sum()
    kernel = g.unsqueeze(0) * g.unsqueeze(1)            # (k, k)
    return kernel.unsqueeze(0).unsqueeze(0).expand(3, 1, k, k).contiguous()


def eot_augment_differentiable(composite: torch.Tensor) -> torch.Tensor:
    """
    Fully differentiable EOT augmentation.

    All geometric transforms (scale, rotation, perspective) are applied via
    torch.nn.functional.grid_sample so gradients flow back to patch pixels.
    Photometric transforms are pure tensor ops.  JPEG compression uses a
    straight-through estimator so the forward YOLO pass sees a realistically
    compressed image while gradients still flow through the patch.

    Replaces: eot_augment_tensor(composite.detach()) + re-stamp.
    """
    t = composite                          # (1, 3, H, W) float32 [0, 1]
    _, _, H, W = t.shape
    device = t.device

    # ── Photometric (tensor ops — differentiable) ─────────────────────────
    # 1. Brightness + contrast
    alpha = 1.0 + random.uniform(-EOT_BRIGHTNESS, EOT_BRIGHTNESS)
    beta  = random.uniform(-40.0 / 255.0, 40.0 / 255.0)
    t = torch.clamp(t * alpha + beta, 0.0, 1.0)

    # 2. Gamma correction
    gamma = random.uniform(*EOT_GAMMA_RANGE)
    t = torch.clamp(t.pow(gamma), 0.0, 1.0)

    # 3. Per-channel colour jitter
    channels = list(t.unbind(dim=1))       # [ (1, H, W) × 3 ]
    for c in range(3):
        shift = random.uniform(-EOT_COLOR_JITTER, EOT_COLOR_JITTER)
        channels[c] = torch.clamp(channels[c] + shift, 0.0, 1.0)
    t = torch.stack(channels, dim=1)

    # 4. Gaussian blur (depthwise conv — differentiable)
    k = random.choice([1, 1, 3, 3,
                       EOT_BLUR_MAX if EOT_BLUR_MAX % 2 == 1 else EOT_BLUR_MAX + 1])
    if k > 1:
        kern = _gaussian_kernel_2d(k, device)
        t = torch.nn.functional.conv2d(t, kern, padding=k // 2, groups=3)

    # ── Geometric (grid_sample — differentiable) ──────────────────────────
    # 5. Scale + rotation as one affine grid
    sc        = random.uniform(*EOT_SCALE_RANGE)
    angle_rad = math.radians(random.uniform(-EOT_ROT_RANGE, EOT_ROT_RANGE))
    ca = math.cos(angle_rad) / sc
    sa = math.sin(angle_rad) / sc
    theta = torch.tensor(
        [[ca, -sa, 0.0],
         [sa,  ca, 0.0]],
        dtype=torch.float32, device=device,
    ).unsqueeze(0)
    grid = torch.nn.functional.affine_grid(theta, t.shape, align_corners=False)
    t = torch.nn.functional.grid_sample(
        t, grid, mode='bilinear', padding_mode='zeros', align_corners=False
    )

    # 6. Perspective warp — build sampling grid from inverted homography
    jx = EOT_PERSP_JITTER * W
    jy = EOT_PERSP_JITTER * H
    src_pts = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    dst_pts = np.float32([
        [random.uniform(0, jx),     random.uniform(0, jy)],
        [random.uniform(W - jx, W), random.uniform(0, jy)],
        [random.uniform(W - jx, W), random.uniform(H - jy, H)],
        [random.uniform(0, jx),     random.uniform(H - jy, H)],
    ])
    # Invert: grid_sample needs input coords for each output pixel
    H_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
    ys, xs = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij',
    )
    ones   = torch.ones(H * W, dtype=torch.float32, device=device)
    coords = torch.stack([xs.reshape(-1), ys.reshape(-1), ones], dim=0)  # (3, N)
    H_inv_t = torch.tensor(H_inv, dtype=torch.float32, device=device)
    mapped  = H_inv_t @ coords                                           # (3, N)
    w_hom   = mapped[2].clamp(min=1e-6)
    gx = (mapped[0] / w_hom) / (W * 0.5) - 1.0
    gy = (mapped[1] / w_hom) / (H * 0.5) - 1.0
    persp_grid = torch.stack([gx, gy], dim=-1).reshape(1, H, W, 2)
    t = torch.nn.functional.grid_sample(
        t, persp_grid, mode='bilinear', padding_mode='zeros', align_corners=False
    )

    # ── Noise & shadow ────────────────────────────────────────────────────
    # 7. Random shadow strip (tensor mask — differentiable)
    if random.random() < EOT_SHADOW_PROB:
        intensity = random.uniform(0.3, 0.7)
        shadow = torch.ones(1, 1, H, W, dtype=torch.float32, device=device)
        if random.random() < 0.5:
            y1 = random.randint(0, H // 2)
            y2 = random.randint(H // 2, H)
            shadow[:, :, y1:y2, :] = intensity
        else:
            x1 = random.randint(0, W // 2)
            x2 = random.randint(W // 2, W)
            shadow[:, :, :, x1:x2] = intensity
        t = t * shadow

    # 8. Gaussian print noise (additive — differentiable)
    if EOT_PRINT_NOISE > 0:
        noise_std = random.uniform(0, EOT_PRINT_NOISE / 255.0)
        t = torch.clamp(t + torch.randn_like(t) * noise_std, 0.0, 1.0)

    # 9. JPEG simulation — straight-through estimator:
    #    forward pass: YOLO sees the JPEG-compressed image (realistic).
    #    backward pass: gradients flow through the pre-JPEG differentiable t.
    quality = random.randint(EOT_JPEG_QUALITY[0], EOT_JPEG_QUALITY[1])
    arr = (
        t.detach().squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
    ).clip(0, 255).astype(np.uint8)
    _, enc    = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    arr_dec   = cv2.imdecode(enc, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
    t_jpeg    = torch.from_numpy(arr_dec).permute(2, 0, 1).unsqueeze(0).to(device)
    t = t_jpeg.detach() + (t - t.detach())   # STE: forward=jpeg, backward=clean

    return t


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

_SHAPE_PRINTED = False


def forward_person_loss(
    torch_model: torch.nn.Module,
    img_t: torch.Tensor,
    top_k: int,
    device: torch.device,
    bbox: tuple[int, int, int, int] | None = None,
    anchor_centers: torch.Tensor | None = None,
    iou_sigma: float = 0.5,
) -> torch.Tensor:
    """
    Forward pass returning scalar loss = mean(top-k person confs).

    Mean-only loss spreads the gradient evenly across all high-confidence
    anchors, driving consistent confidence suppression across every frame
    rather than hunting for single lucky knockouts (FNs).
    Output layout: (1, 4+nc, 8400)  — channels-first (Ultralytics 8.x).

    IoU-guided mode (bbox + anchor_centers provided):
        Each anchor's person score is multiplied by a Gaussian proximity
        weight relative to the detected person bbox before top-k selection.
        This focuses the gradient on anchors that genuinely overlap the
        person, filtering out background / noise anchor activations that
        would otherwise dilute the adversarial signal.
    """
    global _SHAPE_PRINTED
    pred = torch_model(img_t.to(device))
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    if not _SHAPE_PRINTED:
        print(f"[DEBUG] raw pred shape: {tuple(pred.shape)}")
        _SHAPE_PRINTED = True
    person_scores = pred[0, PERSON_COL_IDX, :]          # (N_anchors,)
    if bbox is not None and anchor_centers is not None:
        iou_w = bbox_iou_weights(bbox, anchor_centers, iou_sigma).to(device)
        person_scores = person_scores * iou_w
    k = min(top_k, person_scores.shape[0])
    top_scores = torch.topk(person_scores, k).values
    return top_scores.mean()


# ---------------------------------------------------------------------------
# NPS + TV losses  (Thys et al. 2019 / Sharif et al. 2016)
# ---------------------------------------------------------------------------

def load_printable_colors(path: str, device: torch.device) -> torch.Tensor:
    """
    Load the printable colour set from a text file.
    Each line: R G B  (float [0,1]).
    Returns a (N, 3) float32 tensor on `device`.
    """
    colors = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            r, g, b = map(float, line.split())
            colors.append([r, g, b])
    return torch.tensor(colors, dtype=torch.float32, device=device)  # (N, 3)


def nps_loss(patch: torch.Tensor, printable_colors: torch.Tensor) -> torch.Tensor:
    """
    Non-Printability Score — Sharif et al. (2016).

    L_nps = sum over all pixels of: min over printable colours of |pixel - colour|_1

    Encourages each patch pixel to be as close as possible to at least one
    printer-reproducible colour.  Low NPS → more physically printable patch.

    patch            : (1, 3, H, W)  — requires_grad=True
    printable_colors : (N_colors, 3)
    """
    # Reshape patch to (H*W, 3)
    p = patch.squeeze(0).permute(1, 2, 0).reshape(-1, 3)   # (H*W, 3)
    # Pairwise L1 distances: (H*W, N_colors)
    dists = (p.unsqueeze(1) - printable_colors.unsqueeze(0)).abs().sum(dim=2)
    # Minimum distance to any printable colour for each pixel — mean over pixels
    # so that the scale is independent of patch resolution and comparable to L_obj
    return dists.min(dim=1).values.mean()


def tv_loss(patch: torch.Tensor) -> torch.Tensor:
    """
    Total Variation loss — Thys et al. (2019) / Sharif et al. (2016).

    L_tv = sum_{i,j} [ (p_{i,j} - p_{i+1,j})^2 + (p_{i,j} - p_{i,j+1})^2 ]

    Penalises abrupt pixel-to-pixel changes; encourages smooth patches that
    survive JPEG compression, print-then-photograph, and camera capture.

    patch : (1, 3, H, W)  — requires_grad=True
    """
    dh = patch[:, :, 1:, :] - patch[:, :, :-1, :]   # vertical differences
    dw = patch[:, :, :, 1:] - patch[:, :, :, :-1]   # horizontal differences
    n_pixels = patch.shape[2] * patch.shape[3]
    return ((dh ** 2).sum() + (dw ** 2).sum()) / n_pixels


# ---------------------------------------------------------------------------
# Letter shape loss  (optional visible character embedding)
# ---------------------------------------------------------------------------

def generate_letter_mask(letter: str, patch_size: int, device: torch.device) -> torch.Tensor:
    """
    Render `letter` as a white-on-black grayscale mask using OpenCV.
    The letter is scaled to fill ~75% of the patch height and centred.

    Returns (1, 1, patch_size, patch_size) float32 in [0, 1] on `device`.
    White pixels (≈1) mark the letter foreground; black pixels (0) mark background.
    """
    canvas = np.zeros((patch_size, patch_size), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_DUPLEX
    scale, thickness = 1.0, 2
    for s_int in range(5, 400):
        s = s_int * 0.1
        th = max(2, int(s * 3))
        (_, ch), _ = cv2.getTextSize(letter, font, s, thickness=th)
        if ch >= patch_size * 0.72:
            scale, thickness = s, th
            break
    (tw, th), _ = cv2.getTextSize(letter, font, scale, thickness)
    x = max(0, (patch_size - tw) // 2)
    y = min(patch_size - 4, (patch_size + th) // 2)
    cv2.putText(canvas, letter, (x, y), font, scale, 255, thickness, cv2.LINE_AA)
    arr = canvas.astype(np.float32) / 255.0
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    return t.to(device)


def letter_shape_loss(patch: torch.Tensor, letter_mask: torch.Tensor) -> torch.Tensor:
    """
    Soft luminance-contrast loss that imprints a letterform into the patch.

    letter_mask : (1, 1, H, W) float in [0, 1] — 1 = letter pixels, 0 = background.
    patch       : (1, 3, H, W) float in [0, 1] — requires_grad=True.

    Computes MSE between the patch ITU-R BT.601 luminance and the mask:
      • Letter pixels  (mask ≈ 1) are pushed toward bright luminance.
      • Background pixels (mask ≈ 0) are pushed toward dark luminance.

    This creates luminance contrast that makes the letter readable without
    constraining any hue — colour choice is left to the adversarial loss.
    """
    lum = 0.299 * patch[:, 0:1] + 0.587 * patch[:, 1:2] + 0.114 * patch[:, 2:3]
    return torch.nn.functional.mse_loss(lum, letter_mask)


# ---------------------------------------------------------------------------
# Perlin noise helper (pure numpy, no extra dependencies)
# ---------------------------------------------------------------------------

def _perlin_noise_2d(
    size: int,
    octaves: int = 6,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a (size, size) float32 array of multi-octave Perlin noise in [0,1].
    Uses classic fade/lerp/gradient logic with a seeded permutation table.
    """
    rng = np.random.RandomState(seed)

    def fade(t: np.ndarray) -> np.ndarray:
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    def lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
        return a + t * (b - a)

    # 8-direction gradient vectors
    _GRADS = np.array(
        [[1, 1], [-1, 1], [1, -1], [-1, -1],
         [1, 0], [-1, 0], [0, 1],  [0, -1]],
        dtype=np.float32,
    )

    def gradient_dot(h: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
        g = _GRADS[h % 8]
        return g[..., 0] * dx + g[..., 1] * dy

    result    = np.zeros((size, size), dtype=np.float32)
    amplitude = 1.0
    frequency = 1.0
    max_val   = 0.0

    for _ in range(octaves):
        p = rng.permutation(256).astype(np.int32)
        p = np.concatenate([p, p])           # doubled for wrap-around indexing

        # Fractional coordinates scaled by current frequency
        lin = np.linspace(0.0, frequency, size, endpoint=False, dtype=np.float32)
        gx, gy = np.meshgrid(lin, lin)       # (size, size)

        xi = gx.astype(np.int32) & 255
        yi = gy.astype(np.int32) & 255
        xf = gx - gx.astype(np.int32)
        yf = gy - gy.astype(np.int32)

        u = fade(xf)
        v = fade(yf)

        n00 = gradient_dot(p[p[xi]     + yi    ], xf,     yf    )
        n10 = gradient_dot(p[p[xi + 1] + yi    ], xf - 1, yf    )
        n01 = gradient_dot(p[p[xi]     + yi + 1], xf,     yf - 1)
        n11 = gradient_dot(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)

        x1 = lerp(n00, n10, u)
        x2 = lerp(n01, n11, u)

        result  += amplitude * lerp(x1, x2, v)
        max_val += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    # Normalise to [0, 1]
    result /= max_val
    lo, hi  = result.min(), result.max()
    result  = (result - lo) / (hi - lo + 1e-8)
    return result


# ---------------------------------------------------------------------------
# Random torso-band placement
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Patch initialisations
# ---------------------------------------------------------------------------

def init_patch(mode: str, patch_size: int, device: torch.device) -> torch.Tensor:
    """
    Return a (1, 3, patch_size, patch_size) float32 tensor in [0, 1].

    Modes
    -----
    uniform      : each pixel drawn from U(0,1)  [current baseline]
    gaussian     : each pixel ~ clip(N(0.5, 0.2), 0, 1)
    checkerboard : hard black/white alternating squares (8px cells)
    stripes      : vertical stripes alternating 0 and 1 (16px bands)
    salt_pepper  : pixels randomly set to 0 or 1, rest 0.5 grey
    gray         : flat 0.447 grey (ImageNet mean) — pure gradient signal
    """
    ps = patch_size
    if mode == "uniform":
        t = torch.rand(1, 3, ps, ps)

    elif mode == "gaussian":
        t = torch.randn(1, 3, ps, ps) * 0.2 + 0.5
        t = t.clamp(0.0, 1.0)

    elif mode == "checkerboard":
        cell = 8
        arr = np.zeros((ps, ps), dtype=np.float32)
        for r in range(ps):
            for c in range(ps):
                if ((r // cell) + (c // cell)) % 2 == 0:
                    arr[r, c] = 1.0
        arr3 = np.stack([arr, arr, arr], axis=0)  # (3,H,W)
        t = torch.from_numpy(arr3).unsqueeze(0)

    elif mode == "stripes":
        band = 16
        arr = np.zeros((ps, ps), dtype=np.float32)
        for c in range(ps):
            if (c // band) % 2 == 0:
                arr[:, c] = 1.0
        arr3 = np.stack([arr, arr, arr], axis=0)
        t = torch.from_numpy(arr3).unsqueeze(0)

    elif mode == "salt_pepper":
        arr = np.full((ps, ps), 0.5, dtype=np.float32)
        mask = np.random.rand(ps, ps)
        arr[mask < 0.15] = 0.0   # pepper
        arr[mask > 0.85] = 1.0   # salt
        arr3 = np.stack([arr, arr, arr], axis=0)
        t = torch.from_numpy(arr3).unsqueeze(0)

    elif mode == "gray":
        t = torch.full((1, 3, ps, ps), 0.447)   # ≈ ImageNet mean

    elif mode == "blocky":
        # High-density 8×8 block noise: every block gets an independent
        # extreme colour per channel (0 or 1), maximising spatial contrast
        # at a scale that matches YOLOv8n's early convolutional receptive fields.
        rng  = np.random.RandomState(42)
        arr  = np.zeros((ps, ps, 3), dtype=np.float32)
        cell = 8
        for r in range(0, ps, cell):
            for c in range(0, ps, cell):
                colour = rng.randint(0, 2, size=3).astype(np.float32)  # [0,1]^3
                arr[r:r + cell, c:c + cell] = colour
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()

    elif mode == "perlin":
        # Multi-octave Perlin noise (6 octaves, persistence 0.5, lacunarity 2).
        # Three independent noise fields are generated with different seeds so
        # each RGB channel has its own coherent low-frequency structure.
        channels = [
            _perlin_noise_2d(ps, octaves=6, persistence=0.5,
                             lacunarity=2.0, seed=42 + ch)
            for ch in range(3)
        ]
        arr = np.stack(channels, axis=0)          # (3, H, W)
        t   = torch.from_numpy(arr).unsqueeze(0).float()

    else:
        raise ValueError(f"Unknown init mode: {mode}")

    return t.float().to(device)


def random_torso_placement(img_size: int, patch_size: int) -> tuple[int, int]:
    """
    Sample (row, col) within the upper-torso band (25–55 % of image height)
    and a ±10 % horizontal jitter around centre.
    """
    row_min = int(img_size * 0.25)
    row_max = max(row_min, int(img_size * 0.55) - patch_size)
    row = random.randint(row_min, row_max) if row_max > row_min else row_min

    col_center = (img_size - patch_size) // 2
    col_jitter = int(img_size * 0.10)
    col = random.randint(
        max(0, col_center - col_jitter),
        min(img_size - patch_size, col_center + col_jitter),
    )
    return row, col


# Maximum fraction of bbox height the hat patch is allowed to reach DOWN from
# the top of the bbox.  0.12 keeps the patch firmly in the crown zone — the
# face starts at roughly 10–15 % of full-body bbox height, so this hard cap
# ensures we are NOT occluding the face even at worst-case bbox aspect ratios.
_HAT_FACE_GUARD = 0.12


def hat_crown_placement(
    bbox: tuple[int, int, int, int],
    img_size: int,
    hat_fraction: float = 0.08,
) -> tuple[int, int, int]:
    """
    Derive (row, col, crown_size) for the hat/crown patch from a person bbox.

    crown_size = hat_fraction × bbox_height, clamped to [16, bbox_h * _HAT_FACE_GUARD].
    The patch is centred horizontally on the bbox and anchored to the very TOP
    of the bbox so it sits on the crown of the head.

    Anti-cheating guarantee
    -----------------------
    We enforce that the patch BOTTOM edge never exceeds y1 + _HAT_FACE_GUARD * bbox_h.
    The face region starts at approximately 10–15 % below y1 in a full-body
    bounding box, so this clamp keeps the patch strictly above the face.
    Any call that would violate this is silently clamped — no gradient flows
    through facial pixels.
    """
    x1, y1, x2, y2 = bbox
    cx      = (x1 + x2) // 2
    bbox_h  = max(y2 - y1, 1)

    # Size: hat_fraction of bbox height, hard-clamped so bottom < face zone
    hat_fraction = min(hat_fraction, _HAT_FACE_GUARD)
    crown_size   = max(16, int(bbox_h * hat_fraction))

    # Hard clamp: bottom of patch must stay above face guard line
    face_guard_row = y1 + int(bbox_h * _HAT_FACE_GUARD)
    crown_size     = min(crown_size, max(16, face_guard_row - y1))

    # Row: top of bbox with small upward jitter (patch may go slightly above y1)
    jitter = max(1, crown_size // 4)
    row    = y1 - random.randint(0, jitter)          # can sit slightly above bbox top
    row    = max(0, min(row, img_size - crown_size))

    # Belt-and-braces: re-enforce the bottom-edge face guard after jitter
    if row + crown_size > face_guard_row:
        row = max(0, face_guard_row - crown_size)

    col = cx - crown_size // 2
    col = max(0, min(col, img_size - crown_size))

    return row, col, crown_size


def bbox_guided_placement(
    bbox: tuple[int, int, int, int],
    patch_size: int,
    img_size: int,
    patch_fraction: float = 0.35,
) -> tuple[int, int, int]:
    """
    Derive (row, col, composite_size) from a detected person bounding box.

    composite_size = patch_fraction × bbox_height, clamped to [32, img_size//2].
    This ensures the patch is always proportionally sized to the person regardless
    of viewing distance — preventing the optimiser from exploiting occlusion.

    The anchor sits at 30% down from the top of the bbox (upper chest / t-shirt
    print area — three-quarters up from the bottom).  ±20% bbox-height jitter
    is applied each step so the patch generalises across realistic garment shift.
    """
    x1, y1, x2, y2 = bbox
    cx  = (x1 + x2) // 2
    bbox_h = max(y2 - y1, 1)
    # Upper chest anchor: 30% down from top of bbox
    cy  = y1 + int(bbox_h * 0.30)

    composite_size = int(bbox_h * patch_fraction)
    composite_size = max(32, min(composite_size, img_size // 2))

    jitter = int(bbox_h * 0.20)
    row = cy - composite_size // 2 + random.randint(-jitter, jitter)
    col = cx - composite_size // 2 + random.randint(-jitter, jitter)

    row = max(0, min(row, img_size - composite_size))
    col = max(0, min(col, img_size - composite_size))

    return row, col, composite_size


# ---------------------------------------------------------------------------
# Clean-image confidence evaluator  (used for baseline + post-training report)
# ---------------------------------------------------------------------------

def save_eval_samples(
    yolo_wrapper,
    host_pool_bgr: list,
    host_bboxes: list,
    baseline_anchor_boxes: list,
    patch_t,
    hat_patch_t,
    patch_size: int,
    patch_fraction: float,
    hat_fraction: float,
    do_bbox_placement: bool,
    out_dir: Path,
    n_samples: int = 5,
    conf_thresh: float = 0.01,
    iou_match_thresh: float = 0.10,
    v_shape_crop: np.ndarray | None = None,
    v_centre_dip_row: int = 0,
    v_width_frac: float = 0.85,
    host_chins: list | None = None,
    chin_fallback_frac: float = 0.15,
) -> None:
    """
    Save n_samples annotated evaluation images to out_dir/eval_samples/.

    Each saved image shows:
      - The host with the best patch composited at its deterministic placement
      - BLUE box  : baseline anchor (where the person was detected clean)
      - GREEN box : post-patch detection that overlaps the anchor (if any)
                    labelled with its confidence score
      - RED box   : where the patch was placed on the torso
      - If no overlapping box is found, a "SUPPRESSED" label is drawn instead.

    Indices are spread evenly across the host pool so the sample is representative.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n     = len(host_pool_bgr)
    idxs  = [int(round(i * (n - 1) / max(n_samples - 1, 1))) for i in range(n_samples)]
    idxs  = list(dict.fromkeys(idxs))[:n_samples]   # deduplicate while preserving order

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness  = 2

    for rank, idx in enumerate(idxs):
        img = host_pool_bgr[idx].copy()
        bb  = host_bboxes[idx] if (do_bbox_placement and idx < len(host_bboxes)) else None

        # ---- Composite patch ------------------------------------------------
        p_np  = patch_t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        p_bgr_full = cv2.cvtColor((p_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        if v_shape_crop is not None and bb is not None and do_bbox_placement:
            chin_row, _ = (
                host_chins[idx]
                if (host_chins is not None and idx < len(host_chins) and host_chins[idx] is not None)
                else detect_chin_row(host_pool_bgr[idx], bb, chin_fallback_frac)
            )
            v_row0, v_col0, v_w, v_h, v_mask = v_shape_placement(
                bb, chin_row, v_width_frac, IMG_SIZE, v_shape_crop, v_centre_dip_row
            )
            img = _apply_v_shape_bgr(img, p_bgr_full, v_row0, v_col0, v_w, v_h, v_mask)
            p_row, p_col, comp_size = max(0, v_row0), v_col0, v_w
        elif bb is not None and do_bbox_placement:
            x1, y1, x2, y2 = bb
            cx        = (x1 + x2) // 2
            cy        = (y1 + y2) // 2
            bbox_h    = max(y2 - y1, 1)
            comp_size = max(32, min(int(bbox_h * patch_fraction), IMG_SIZE // 2))
            p_row = max(0, min(cy - comp_size // 2, IMG_SIZE - comp_size))
            p_col = max(0, min(cx - comp_size // 2, IMG_SIZE - comp_size))
            p_bgr = cv2.resize(p_bgr_full, (comp_size, comp_size))
            img[p_row:p_row + comp_size, p_col:p_col + comp_size] = p_bgr
        else:
            p_row     = int(IMG_SIZE * 0.30)
            p_col     = (IMG_SIZE - patch_size) // 2
            comp_size = patch_size
            p_bgr = cv2.resize(p_bgr_full, (comp_size, comp_size))
            img[p_row:p_row + comp_size, p_col:p_col + comp_size] = p_bgr

        if hat_patch_t is not None and bb is not None and do_bbox_placement:
            h_row, h_col, h_size = hat_crown_placement(
                bb, IMG_SIZE, min(hat_fraction, _HAT_FACE_GUARD)
            )
            h_np  = hat_patch_t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            h_bgr = cv2.cvtColor((h_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            h_bgr = cv2.resize(h_bgr, (h_size, h_size))
            img[h_row:h_row + h_size, h_col:h_col + h_size] = h_bgr

        # ---- Run detection --------------------------------------------------
        with torch.no_grad():
            results = yolo_wrapper.predict(
                source=img, conf=conf_thresh, classes=[PERSON_CLASS], verbose=False
            )
        boxes     = results[0].boxes
        det_xyxy  = boxes.xyxy.cpu().numpy()  if (boxes is not None and len(boxes) > 0) else []
        det_confs = boxes.conf.cpu().numpy()  if (boxes is not None and len(boxes) > 0) else []

        anchor = baseline_anchor_boxes[idx] if idx < len(baseline_anchor_boxes) else None

        # ---- Draw patch placement box (RED) ---------------------------------
        cv2.rectangle(img,
                      (p_col, p_row), (p_col + comp_size, p_row + comp_size),
                      (0, 0, 220), 2)
        cv2.putText(img, "patch", (p_col + 4, p_row - 6),
                    font, font_scale * 0.8, (0, 0, 220), thickness - 1, cv2.LINE_AA)

        # ---- Draw baseline anchor (BLUE) ------------------------------------
        if anchor is not None:
            ax1, ay1, ax2, ay2 = anchor
            cv2.rectangle(img, (ax1, ay1), (ax2, ay2), (220, 100, 0), 2)
            cv2.putText(img, "baseline", (ax1 + 4, ay1 - 6),
                        font, font_scale * 0.8, (220, 100, 0), thickness - 1, cv2.LINE_AA)

        # ---- Find highest-confidence post-patch detection & draw (GREEN) -----
        # Always show the top-conf box regardless of position — suppression
        # is only declared when the detector returns no boxes at all.  This
        # lets us see whether the detector has shifted to a partial hit (face,
        # legs, etc.) rather than masking that as a "successful" suppression.
        matched_conf = None
        if len(det_xyxy) > 0:
            best_j       = int(np.argmax(det_confs))
            matched_conf = float(det_confs[best_j])
            dx1, dy1, dx2, dy2 = det_xyxy[best_j].astype(int)
            cv2.rectangle(img, (dx1, dy1), (dx2, dy2), (0, 200, 0), 2)
            cv2.putText(img, f"conf {matched_conf:.3f}", (dx1 + 4, dy2 + 18),
                        font, font_scale, (0, 200, 0), thickness, cv2.LINE_AA)

        if matched_conf is None:
            cv2.putText(img, "SUPPRESSED",
                        (IMG_SIZE // 2 - 80, IMG_SIZE // 2),
                        font, 1.0, (0, 200, 0), 2, cv2.LINE_AA)

        # ---- Legend ---------------------------------------------------------
        legend_y = IMG_SIZE - 12
        cv2.putText(img, "RED=patch  BLUE=baseline  GREEN=post-patch detection",
                    (8, legend_y), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "RED=patch  BLUE=baseline  GREEN=post-patch detection",
                    (7, legend_y - 1), font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        fname = out_dir / f"sample_{rank + 1:02d}_host{idx:02d}.jpg"
        cv2.imwrite(str(fname), img)

    print(f"[INFO] Eval samples saved → {out_dir}  ({len(idxs)} images)")


def _box_iou(a: tuple, b: tuple) -> float:
    """
    Compute IoU between two (x1, y1, x2, y2) boxes.
    Returns 0.0 if either box has zero area.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1);  iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2);  iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def clean_eval_confidence(
    yolo_wrapper,
    host_pool_bgr: list,
    host_bboxes: list,
    patch_t,
    hat_patch_t,
    patch_size: int,
    patch_fraction: float,
    hat_fraction: float,
    do_bbox_placement: bool,
    device,
    conf_thresh: float = 0.01,
    anchor_boxes: list | None = None,
    iou_match_thresh: float = 0.10,
    v_shape_crop: np.ndarray | None = None,
    v_centre_dip_row: int = 0,
    v_width_frac: float = 0.85,
    host_chins: list | None = None,
    chin_fallback_frac: float = 0.15,
) -> tuple[float, list]:
    """
    Measure the average person-detection confidence across all host images.

    Baseline pass  (patch_t=None, anchor_boxes=None):
        Takes the highest-confidence person box per image as the reference.
        Returns (mean_conf, detected_boxes) where detected_boxes[i] is the
        (x1,y1,x2,y2) of the chosen box, or None if no detection.

    Adversarial pass (patch_t given, anchor_boxes=<baseline list>):
        Composites the patch, runs predict, then for each image finds the
        detected box with the highest IoU vs the baseline anchor box.
        If no box overlaps the anchor above iou_match_thresh, the person is
        considered fully suppressed and the image contributes 0.0.
        This guarantees baseline and post-patch always measure the same subject.

    Returns (mean_confidence, per_image_detected_boxes).
    """
    confs        = []
    found_boxes  = []   # (x1,y1,x2,y2) or None per image

    for i, host_bgr in enumerate(host_pool_bgr):
        img = host_bgr.copy()

        if patch_t is not None:
            bb = host_bboxes[i] if (do_bbox_placement and i < len(host_bboxes)) else None
            p_np      = patch_t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            p_bgr_full = cv2.cvtColor((p_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

            if v_shape_crop is not None and bb is not None and do_bbox_placement:
                chin_row, _ = (
                    host_chins[i]
                    if (host_chins is not None and i < len(host_chins) and host_chins[i] is not None)
                    else detect_chin_row(host_bgr, bb, chin_fallback_frac)
                )
                v_row0, v_col0, v_w, v_h, v_mask = v_shape_placement(
                    bb, chin_row, v_width_frac, IMG_SIZE, v_shape_crop, v_centre_dip_row
                )
                img = _apply_v_shape_bgr(img, p_bgr_full, v_row0, v_col0, v_w, v_h, v_mask)
            elif bb is not None and do_bbox_placement:
                x1, y1, x2, y2 = bb
                cx        = (x1 + x2) // 2
                cy        = (y1 + y2) // 2
                bbox_h    = max(y2 - y1, 1)
                comp_size = max(32, min(int(bbox_h * patch_fraction), IMG_SIZE // 2))
                row = max(0, min(cy - comp_size // 2, IMG_SIZE - comp_size))
                col = max(0, min(cx - comp_size // 2, IMG_SIZE - comp_size))
                p_bgr = cv2.resize(p_bgr_full, (comp_size, comp_size))
                img[row:row + comp_size, col:col + comp_size] = p_bgr
            else:
                row       = int(IMG_SIZE * 0.30)
                col       = (IMG_SIZE - patch_size) // 2
                comp_size = patch_size
                p_bgr = cv2.resize(p_bgr_full, (comp_size, comp_size))
                img[row:row + comp_size, col:col + comp_size] = p_bgr

            if hat_patch_t is not None and bb is not None and do_bbox_placement:
                h_row, h_col, h_size = hat_crown_placement(
                    bb, IMG_SIZE, min(hat_fraction, _HAT_FACE_GUARD)
                )
                h_np  = hat_patch_t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                h_bgr = cv2.cvtColor((h_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                h_bgr = cv2.resize(h_bgr, (h_size, h_size))
                img[h_row:h_row + h_size, h_col:h_col + h_size] = h_bgr

        with torch.no_grad():
            results = yolo_wrapper.predict(
                source=img, conf=conf_thresh, classes=[PERSON_CLASS], verbose=False
            )
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            confs.append(0.0)
            found_boxes.append(None)
            continue

        det_xyxy  = boxes.xyxy.cpu().numpy()   # (N, 4)
        det_confs = boxes.conf.cpu().numpy()   # (N,)

        if anchor_boxes is not None and i < len(anchor_boxes) and anchor_boxes[i] is not None:
            # Adversarial pass: find box that best overlaps the baseline anchor
            anchor = anchor_boxes[i]
            ious   = [_box_iou(anchor, tuple(det_xyxy[j].astype(int))) for j in range(len(det_xyxy))]
            best_j = int(np.argmax(ious))
            if ious[best_j] >= iou_match_thresh:
                confs.append(float(det_confs[best_j]))
                found_boxes.append(tuple(det_xyxy[best_j].astype(int)))
            else:
                # No box overlaps the known person location — fully suppressed
                confs.append(0.0)
                found_boxes.append(None)
        else:
            # Baseline pass (or no anchor): take the highest-confidence box
            best_j = int(np.argmax(det_confs))
            confs.append(float(det_confs[best_j]))
            found_boxes.append(tuple(det_xyxy[best_j].astype(int)))

    return (float(np.mean(confs)) if confs else 0.0), found_boxes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    patch_size = args.patch_size
    out_path   = Path(args.out) if args.out else PROJECT_ROOT / "patterns" / f"patch_{patch_size}_{args.init}.png"

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[INFO] Device     : {device}")
    print(f"[INFO] Weights    : {args.model}")
    print(f"[INFO] Patch size : {patch_size}×{patch_size}")
    print(f"[INFO] Steps      : {args.steps}  |  LR: {args.lr}  |  ε: {args.eps}")
    print(f"[INFO] EOT        : {'OFF' if args.no_eot else 'ON'}")
    print(f"[INFO] Init noise : {args.init}")
    print(f"[INFO] Loss mode  : mean (confidence suppression)")
    print(f"[INFO] Batch size : {args.batch_size} images/step")
    print(f"[INFO] NPS α      : {args.alpha}  |  TV β : {args.beta}")
    print(f"[INFO] BBox placement: {'ON  (patch fraction=' + str(args.patch_fraction) + ')' if args.bbox_placement else 'OFF (torso-band fallback)'}")
    if args.hat_patch:
        hf = min(args.hat_fraction, _HAT_FACE_GUARD)
        print(f"[INFO] Hat/crown patch: ON  (hat fraction={hf}, face guard≤{_HAT_FACE_GUARD} of bbox h)")
    if args.target_image:
        print(f"[INFO] Style target: {args.target_image}  |  weight γ : {args.style_weight}")
    if args.iou_loss:
        print(f"[INFO] IoU-guided loss  : ON  (sigma={args.iou_sigma})")
    if args.hard_mining:
        print(f"[INFO] Hard-example mine: ON  (temp={args.hard_temp}, EMA α=0.1)")
    if args.letter:
        print(f"[INFO] Letter embed    : '{args.letter}'  (weight={args.letter_weight})")

    # ------------------------------------------------------------------
    # 1. Load model — freeze all weights
    # ------------------------------------------------------------------
    yolo = YOLO(args.model)
    torch_model: torch.nn.Module = yolo.model
    torch_model.eval().to(device)
    for p in torch_model.parameters():
        p.requires_grad_(False)

    # ------------------------------------------------------------------
    # 2a. Load target style image (optional)
    # ------------------------------------------------------------------
    target_tensor: torch.Tensor | None = None
    if args.target_image:
        tgt_bgr = load_bgr(args.target_image, patch_size)
        target_tensor = preprocess(tgt_bgr).to(device)

    # ------------------------------------------------------------------
    # 2. Load printable colours for NPS loss
    # ------------------------------------------------------------------
    printable_colors: torch.Tensor | None = None
    if args.alpha > 0:
        printable_colors = load_printable_colors(args.printable_colors, device)
        print(f"[INFO] Printable colours: {printable_colors.shape[0]} loaded from {args.printable_colors}")

    # Letter shape mask — rendered once, reused every training step.
    letter_mask: torch.Tensor | None = None
    if args.letter:
        letter_mask = generate_letter_mask(args.letter, patch_size, device)
        print(f"[INFO] Letter mask     : '{args.letter}' rendered at {patch_size}×{patch_size}px")

    # ------------------------------------------------------------------
    # 3. Build host image pool  (keep bgr + tensor in sync by index)
    # ------------------------------------------------------------------
    host_pool_bgr = load_host_pool(args.hosts_dir, args.host)
    host_pool_t   = [preprocess(h).to(device) for h in host_pool_bgr]

    # Hat-patch needs bbox-placement — resolve BEFORE the detection pass below.
    if args.hat_patch and not args.bbox_placement:
        print("[WARN] --hat-patch requires --bbox-placement. Enabling automatically.")
        args.bbox_placement = True

    if args.v_shape and not args.bbox_placement:
        print("[WARN] --v-shape requires --bbox-placement. Enabling automatically.")
        args.bbox_placement = True

    # Pre-detect person bboxes for all host images when bbox-placement is on.
    # Done once here so we don't re-run detection every batch sample.
    host_bboxes: list[tuple[int, int, int, int] | None] = []
    if args.bbox_placement:
        print("[INFO] Pre-detecting person bboxes in host pool …")
        detected_count = 0
        for bgr in host_pool_bgr:
            bb = detect_person_bbox(yolo, bgr)
            host_bboxes.append(bb)
            if bb is not None:
                detected_count += 1
        print(f"[INFO]   {detected_count}/{len(host_pool_bgr)} host images have detectable persons")
        if detected_count == 0:
            print("[WARN] No persons detected in any host image — falling back to torso-band placement.")
            args.bbox_placement = False

    # V-shape: load mask + pre-detect chins + pre-cache per-host placement (done once)
    v_shape_crop: np.ndarray | None = None
    v_centre_dip_row: int = 0
    host_chins: list | None = None
    # host_v_placements[i] = (row0, col0, v_w, v_h, mask_tensor) or None
    host_v_placements: list | None = None
    if args.v_shape:
        v_shape_crop, v_centre_dip_row = load_v_shape_mask(args.v_shape_mask)
        print(f"[INFO] V-shape mask loaded: {v_shape_crop.shape[1]}w × {v_shape_crop.shape[0]}h px  "
              f"(centre dip row={v_centre_dip_row})")
        print("[INFO] Pre-detecting chins + caching V-shape placements …")
        host_chins = []
        host_v_placements = []
        haar_count, v_count = 0, 0
        for bgr, bb in zip(host_pool_bgr, host_bboxes):
            if bb is not None:
                chin_row, method = detect_chin_row(bgr, bb, args.chin_fallback_frac)
                host_chins.append((chin_row, method))
                if method == 'haar':
                    haar_count += 1
                row0, col0, v_w, v_h, mask_bin = v_shape_placement(
                    bb, chin_row, args.v_width_frac, IMG_SIZE, v_shape_crop, v_centre_dip_row
                )
                # Pre-convert mask to float tensor on device so it's never
                # re-allocated during the training loop
                mask_t = torch.from_numpy(
                    mask_bin.astype(np.float32) / 255.0
                ).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,v_h,v_w)
                host_v_placements.append((row0, col0, v_w, v_h, mask_t))
                v_count += 1
            else:
                host_chins.append(None)
                host_v_placements.append(None)
        print(f"[INFO]   chin detected via Haar: {haar_count}/{len(host_pool_bgr)}")
        print(f"[INFO]   V-shape placements cached: {v_count}/{len(host_pool_bgr)} "
              f"({len(host_pool_bgr) - v_count} hosts fall back to square torso placement)")

    # Re-pin model to device — yolo.predict() calls in detect_person_bbox can
    # silently move model weights back to CPU on MPS/CUDA systems.
    torch_model.eval().to(device)
    for p in torch_model.parameters():
        p.requires_grad_(False)

    # Precompute the 8400 anchor centre grid for IoU-guided loss.
    # Done once here — reused every batch sample without reallocation.
    anchor_centers: torch.Tensor | None = None
    if args.iou_loss:
        anchor_centers = generate_anchor_centers(IMG_SIZE, device)
        print(f"[INFO] IoU anchor grid  : {anchor_centers.shape[0]} centres precomputed")

    # Hard-example mining: per-host EMA confidence tracker.
    # Initialised to 1.0 so all hosts are sampled uniformly at the start;
    # the EMA diverges as the patch learns to fool some hosts faster than others.
    host_conf_ema = np.ones(len(host_pool_t), dtype=np.float32)

    # ------------------------------------------------------------------
    # 4. Initialise patch  (or resume from checkpoint)
    # ------------------------------------------------------------------
    ckpt_path  = out_path.parent / (out_path.stem + "_ckpt.pt")
    start_step = 0
    best_loss  = float("inf")

    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        patch      = ckpt["patch"].to(device).requires_grad_(True)
        start_step = ckpt["step"]
        best_loss  = ckpt["best_loss"]
        best_patch = ckpt["best_patch"].to(device)
        print(f"[INFO] Resumed from checkpoint  step={start_step}  best_loss={best_loss:.6f}")
    else:
        patch = init_patch(args.init, patch_size, device)
        patch.requires_grad_(True)
        best_patch = patch.detach().clone()

    # Hat/crown patch — separate tensor, same size as torso patch.
    # Stored and updated independently so gradients are clean.
    hat_patch: torch.Tensor | None = None
    best_hat_patch: torch.Tensor | None = None
    if args.hat_patch:
        hat_patch = init_patch(args.init, patch_size, device)
        hat_patch.requires_grad_(True)
        best_hat_patch = hat_patch.detach().clone()
        print(f"[INFO] Hat patch initialised  ( {patch_size}×{patch_size}, "
              f"crown zone ≤ {_HAT_FACE_GUARD*100:.0f}% of bbox h from top )")

    # ------------------------------------------------------------------
    # 5. Baseline: average max-confidence person detection, all hosts, no patch
    # ------------------------------------------------------------------
    print(f"[INFO] Computing clean baseline confidence ({len(host_pool_bgr)} hosts, no patch) …")
    torch_model.to("cpu")  # predict() preprocesses on CPU; avoid MPS device mismatch
    clean_baseline, baseline_anchor_boxes = clean_eval_confidence(
        yolo, host_pool_bgr,
        host_bboxes if args.bbox_placement else [],
        patch_t=None, hat_patch_t=None,
        patch_size=patch_size, patch_fraction=args.patch_fraction,
        hat_fraction=args.hat_fraction, do_bbox_placement=args.bbox_placement,
        device=device,
        v_shape_crop=v_shape_crop, v_centre_dip_row=v_centre_dip_row,
        v_width_frac=args.v_width_frac, host_chins=host_chins,
        chin_fallback_frac=args.chin_fallback_frac,
    )
    n_anchors = sum(1 for b in baseline_anchor_boxes if b is not None)
    print(f"[INFO] Clean baseline mean confidence : {clean_baseline:.6f}  ({n_anchors}/{len(host_pool_bgr)} subjects detected)")
    # Re-pin model to training device
    torch_model.eval().to(device)
    for p in torch_model.parameters():
        p.requires_grad_(False)

    # ------------------------------------------------------------------
    # 6. PGD loop with EOT augmentation + multi-host + random placement
    # ------------------------------------------------------------------
    for step in range(start_step + 1, args.steps + 1):
        if patch.grad is not None:
            patch.grad.zero_()
        if hat_patch is not None and hat_patch.grad is not None:
            hat_patch.grad.zero_()

        # (a–e) Mini-batch: average objectness loss over batch_size images
        #       Each sample uses a different host + placement + EOT augmentation
        #       so the gradient points toward suppression that works everywhere.
        obj_losses = []
        for _ in range(args.batch_size):
            # (a) Sample host — hard-mining weighted or uniform.
            #     Hard-mining gives higher probability to hosts where the patch
            #     currently fails (high conf EMA), implementing a curriculum
            #     analogous to the sequence-level loss in paper 2511.16020.
            if args.hard_mining and len(host_pool_t) > 1:
                samp_w = _softmax_weights(host_conf_ema, args.hard_temp)
                idx = random.choices(range(len(host_pool_t)), weights=samp_w, k=1)[0]
            else:
                idx = random.randrange(len(host_pool_t))
            host_t = host_pool_t[idx]

            # (b) Placement — V-shape, bbox-guided, or torso-band fallback
            bb = host_bboxes[idx] if host_bboxes else None
            if args.v_shape and host_v_placements is not None and host_v_placements[idx] is not None:
                row0, col0, v_w, v_h, mask_t = host_v_placements[idx]
                composite = apply_v_shape_patch(
                    host_t, patch, row0, col0, v_w, v_h, mask_t, device
                )
                row, col, composite_size = row0, col0, v_w
            elif args.v_shape and (host_v_placements is None or host_v_placements[idx] is None):
                # No bbox detected for this host — fall back to square torso placement
                row, col = random_torso_placement(IMG_SIZE, patch_size)
                composite_size = patch_size
                composite = apply_patch_resized(host_t, patch, row, col, composite_size)
            elif args.bbox_placement:
                if bb is not None:
                    row, col, composite_size = bbox_guided_placement(
                        bb, patch_size, IMG_SIZE, args.patch_fraction
                    )
                else:
                    row, col = random_torso_placement(IMG_SIZE, patch_size)
                    composite_size = patch_size
                composite = apply_patch_resized(host_t, patch, row, col, composite_size)
            else:
                row, col = random_torso_placement(IMG_SIZE, patch_size)
                composite_size = patch_size
                composite = apply_patch_resized(host_t, patch, row, col, composite_size)

            # (c2) Composite hat/crown patch on top — strictly face-free zone
            if hat_patch is not None and args.bbox_placement and host_bboxes[idx] is not None:
                h_row, h_col, h_size = hat_crown_placement(
                    host_bboxes[idx], IMG_SIZE, min(args.hat_fraction, _HAT_FACE_GUARD)
                )
                composite = apply_patch_resized(composite, hat_patch, h_row, h_col, h_size)

            # (d) EOT: fully differentiable augmentation.
            #     grid_sample carries gradients through scale/rotation/perspective
            #     so the patch learns geometric robustness, not just photometric.
            #     JPEG uses a straight-through estimator.
            #     No re-stamp needed — patch is already baked into composite above.
            if not args.no_eot:
                composite_aug = eot_augment_differentiable(composite)
            else:
                composite_aug = composite

            # (e) Forward — IoU-guided if both flags are set and bbox is available
            _iou_bb = (
                host_bboxes[idx]
                if (args.iou_loss and args.bbox_placement and host_bboxes)
                else None
            )
            sample_loss = forward_person_loss(
                torch_model, composite_aug, args.topk, device,
                bbox=_iou_bb,
                anchor_centers=(anchor_centers if args.iou_loss else None),
                iou_sigma=args.iou_sigma,
            )
            obj_losses.append(sample_loss)
            # Update hard-mining EMA so this host is sampled more often if the
            # patch is still failing on it (loss stays high = high confidence).
            if args.hard_mining:
                host_conf_ema[idx] = (
                    0.9 * host_conf_ema[idx] + 0.1 * sample_loss.detach().item()
                )

        # (e) Mean objectness loss + regularisation (NPS/TV added once, not per sample)
        loss = sum(obj_losses) / len(obj_losses)
        if args.alpha > 0 and printable_colors is not None:
            loss = loss + args.alpha * nps_loss(patch, printable_colors)
            if hat_patch is not None:
                loss = loss + args.alpha * nps_loss(hat_patch, printable_colors)
        if args.beta > 0:
            loss = loss + args.beta * tv_loss(patch)
            if hat_patch is not None:
                loss = loss + args.beta * tv_loss(hat_patch)
        if target_tensor is not None and args.style_weight > 0:
            loss = loss + args.style_weight * content_loss(patch, target_tensor)
        if letter_mask is not None and args.letter_weight > 0:
            loss = loss + args.letter_weight * letter_shape_loss(patch, letter_mask)

        # (f) Backward — gradients flow to both torso and hat patches
        loss.backward()

        # (g) PGD update — torso patch
        with torch.no_grad():
            patch.data -= args.lr * patch.grad.sign()
            if args.eps < 1.0:
                lower = (patch.data - args.eps).clamp(0.0, 1.0)
                upper = (patch.data + args.eps).clamp(0.0, 1.0)
                patch.data.clamp_(lower, upper)
            patch.data.clamp_(0.0, 1.0)

        # (g2) PGD update — hat patch
        if hat_patch is not None and hat_patch.grad is not None:
            with torch.no_grad():
                hat_patch.data -= args.lr * hat_patch.grad.sign()
                if args.eps < 1.0:
                    lower = (hat_patch.data - args.eps).clamp(0.0, 1.0)
                    upper = (hat_patch.data + args.eps).clamp(0.0, 1.0)
                    hat_patch.data.clamp_(lower, upper)
                hat_patch.data.clamp_(0.0, 1.0)

        if loss.item() < best_loss:
            best_loss  = loss.item()
            best_patch = patch.detach().clone()
            if hat_patch is not None:
                best_hat_patch = hat_patch.detach().clone()

        # Periodic checkpoint — allows resuming if interrupted
        if step % args.checkpoint_every == 0:
            ckpt_data = {
                "step":       step,
                "patch":      patch.detach().cpu(),
                "best_patch": best_patch.cpu(),
                "best_loss":  best_loss,
            }
            if hat_patch is not None:
                ckpt_data["hat_patch"]      = hat_patch.detach().cpu()
                ckpt_data["best_hat_patch"] = best_hat_patch.cpu()
            torch.save(ckpt_data, ckpt_path)

        if args.verbose and step % 50 == 0:
            print(f"  Step {step:>5d}/{args.steps}  loss: {loss.item():.6f}  best: {best_loss:.6f}")

    # ------------------------------------------------------------------
    # 6b. Post-training clean eval: same metric with best patch applied
    # ------------------------------------------------------------------
    print(f"[INFO] Computing post-patch confidence ({len(host_pool_bgr)} hosts, best patch applied) …")
    torch_model.to("cpu")  # predict() preprocesses on CPU; avoid MPS device mismatch
    clean_final, _ = clean_eval_confidence(
        yolo, host_pool_bgr,
        host_bboxes if args.bbox_placement else [],
        patch_t=best_patch, hat_patch_t=best_hat_patch,
        patch_size=patch_size, patch_fraction=args.patch_fraction,
        hat_fraction=args.hat_fraction, do_bbox_placement=args.bbox_placement,
        device=device,
        anchor_boxes=baseline_anchor_boxes,
        v_shape_crop=v_shape_crop, v_centre_dip_row=v_centre_dip_row,
        v_width_frac=args.v_width_frac, host_chins=host_chins,
        chin_fallback_frac=args.chin_fallback_frac,
    )
    # Re-pin model after predict() calls
    torch_model.eval().to(device)
    for p in torch_model.parameters():
        p.requires_grad_(False)
    print(f"[INFO] Post-patch mean confidence     : {clean_final:.6f}")
    reduction = (1.0 - clean_final / max(clean_baseline, 1e-9)) * 100
    print(f"[INFO] Confidence reduction : {clean_baseline:.6f} → {clean_final:.6f}  ({reduction:.1f}% suppression)")

    # ------------------------------------------------------------------
    # 7. Save patch + preview
    # ------------------------------------------------------------------
    patch_np  = best_patch.squeeze(0).permute(1, 2, 0).cpu().numpy()
    patch_bgr = cv2.cvtColor((patch_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # --- 7a. Auto-versioned iteration folder ----------------------------
    # Scans patterns/iterations/ for existing iteration_N dirs and creates
    # the next one, preserving every run as evidence.
    iterations_root = PROJECT_ROOT / "patterns" / "iterations"
    iterations_root.mkdir(parents=True, exist_ok=True)
    existing = [d for d in iterations_root.iterdir()
                if d.is_dir() and d.name.startswith("iteration_")]
    nums = []
    for d in existing:
        try:
            nums.append(int(d.name.split("_")[1]))
        except (IndexError, ValueError):
            pass
    next_n   = max(nums, default=0) + 1
    iter_dir = iterations_root / f"iteration_{next_n}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    iter_patch   = iter_dir / out_path.name
    iter_preview = iter_dir / (out_path.stem + "_preview.png")
    iter_params  = iter_dir / "params.txt"

    cv2.imwrite(str(iter_patch), patch_bgr)
    cv2.imwrite(str(iter_preview),
                cv2.resize(patch_bgr, (512, 512), interpolation=cv2.INTER_NEAREST))

    # --- 7b. Eval sample images — patch composited + detection boxes drawn ---
    torch_model.to("cpu")  # predict() preprocesses on CPU; avoid MPS device mismatch
    save_eval_samples(
        yolo_wrapper=yolo,
        host_pool_bgr=host_pool_bgr,
        host_bboxes=host_bboxes if args.bbox_placement else [],
        baseline_anchor_boxes=baseline_anchor_boxes,
        patch_t=best_patch,
        hat_patch_t=best_hat_patch,
        patch_size=patch_size,
        patch_fraction=args.patch_fraction,
        hat_fraction=args.hat_fraction,
        do_bbox_placement=args.bbox_placement,
        out_dir=iter_dir / "eval_samples",
        v_shape_crop=v_shape_crop, v_centre_dip_row=v_centre_dip_row,
        v_width_frac=args.v_width_frac, host_chins=host_chins,
        chin_fallback_frac=args.chin_fallback_frac,
    )
    # Re-pin model after predict() calls in save_eval_samples
    torch_model.eval().to(device)
    for p in torch_model.parameters():
        p.requires_grad_(False)

    # Save hat/crown patch if trained
    if best_hat_patch is not None:
        hat_np  = best_hat_patch.squeeze(0).permute(1, 2, 0).cpu().numpy()
        hat_bgr = cv2.cvtColor((hat_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        hat_patch_path   = iter_dir / (out_path.stem + "_hat.png")
        hat_preview_path = iter_dir / (out_path.stem + "_hat_preview.png")
        cv2.imwrite(str(hat_patch_path), hat_bgr)
        cv2.imwrite(str(hat_preview_path),
                    cv2.resize(hat_bgr, (512, 512), interpolation=cv2.INTER_NEAREST))
        print(f"         hat     : {hat_patch_path.name}  (crown only — face EXCLUDED)")

    # Write training parameters for evidence
    import datetime
    with open(iter_params, "w") as f:
        f.write(f"iteration      : {next_n}\n")
        f.write(f"date           : {datetime.datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"patch_size     : {patch_size}\n")
        f.write(f"steps          : {args.steps}\n")
        f.write(f"lr             : {args.lr}\n")
        f.write(f"eps            : {args.eps}\n")
        f.write(f"init           : {args.init}\n")
        f.write(f"batch_size     : {args.batch_size}\n")
        f.write(f"topk           : {args.topk}\n")
        f.write(f"alpha_nps      : {args.alpha}\n")
        f.write(f"beta_tv        : {args.beta}\n")
        f.write(f"eot            : {not args.no_eot}\n")
        f.write(f"eot_rot_range  : ±{EOT_ROT_RANGE}°\n")
        f.write(f"eot_scale      : {EOT_SCALE_RANGE[0]}–{EOT_SCALE_RANGE[1]}×\n")
        f.write(f"eot_persp      : ±{EOT_PERSP_JITTER * 100:.0f}% corner jitter\n")
        f.write(f"eot_jpeg_q     : {EOT_JPEG_QUALITY[0]}–{EOT_JPEG_QUALITY[1]}\n")
        f.write(f"eot_brightness : ±{EOT_BRIGHTNESS * 100:.0f}%\n")
        f.write(f"eot_gamma      : {EOT_GAMMA_RANGE[0]}–{EOT_GAMMA_RANGE[1]}\n")
        f.write(f"eot_hsv_hue    : ±{EOT_HSV_HUE}°\n")
        f.write(f"eot_hsv_sat    : ±{EOT_HSV_SAT * 100:.0f}%\n")
        f.write(f"eot_shadow     : p={EOT_SHADOW_PROB}\n")
        f.write(f"eot_color_jit  : ±{EOT_COLOR_JITTER * 100:.0f}% per-channel\n")
        f.write(f"eot_print_noise: 0–{EOT_PRINT_NOISE:.0f} std Gaussian\n")
        f.write(f"device         : {device}\n")
        f.write(f"clean_baseline : {clean_baseline:.6f}\n")
        f.write(f"clean_final    : {clean_final:.6f}\n")
        f.write(f"final_loss     : {best_loss:.6f}\n")
        f.write(f"loss_reduction : {reduction:.1f}%\n")
        f.write(f"host_pool_size : {len(host_pool_t)}\n")
        f.write(f"loss_mode      : mean (confidence suppression)\n")
        f.write(f"bbox_placement : {args.bbox_placement}\n")
        f.write(f"patch_fraction : {args.patch_fraction}\n")
        f.write(f"hat_patch      : {args.hat_patch}\n")
        f.write(f"hat_fraction   : {min(args.hat_fraction, _HAT_FACE_GUARD)}\n")
        f.write(f"hat_face_guard : {_HAT_FACE_GUARD} (hard clamp — face never covered)\n")
        f.write(f"iou_loss       : {args.iou_loss}\n")
        f.write(f"iou_sigma      : {args.iou_sigma}\n")
        f.write(f"hard_mining    : {args.hard_mining}\n")
        f.write(f"hard_temp      : {args.hard_temp}\n")
        if args.target_image:
            f.write(f"style_target   : {args.target_image}\n")
            f.write(f"style_weight   : {args.style_weight}\n")
        else:
            f.write(f"style_target   : none\n")
        f.write(f"letter         : {args.letter if args.letter else 'none'}\n")
        f.write(f"letter_weight  : {args.letter_weight}\n")
        f.write(f"v_shape        : {args.v_shape}\n")
        if args.v_shape:
            f.write(f"v_shape_mask   : {args.v_shape_mask}\n")
            f.write(f"v_width_frac   : {args.v_width_frac}\n")
            f.write(f"chin_fallback  : {args.chin_fallback_frac}\n")
            haar_hits = sum(1 for c in (host_chins or []) if c is not None and c[1] == 'haar')
            total_chins = len(host_chins) if host_chins else 0
            f.write(f"chin_haar_hits : {haar_hits}/{total_chins}\n")

    print(f"[INFO] Iteration {next_n} saved → {iter_dir}")
    print(f"         patch   : {iter_patch.name}")
    print(f"         preview : {iter_preview.name}")
    print(f"         params  : {iter_params.name}")

    # --- 7b. Also overwrite flat 'latest' copy so apply_patch.py still works ---
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), patch_bgr)
    preview = out_path.parent / (out_path.stem + "_preview.png")
    cv2.imwrite(str(preview),
                cv2.resize(patch_bgr, (512, 512), interpolation=cv2.INTER_NEAREST))
    print(f"[INFO] Latest copy → {out_path}")

    if best_hat_patch is not None:
        hat_np      = best_hat_patch.squeeze(0).permute(1, 2, 0).cpu().numpy()
        hat_bgr_out = cv2.cvtColor((hat_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        hat_out     = out_path.parent / (out_path.stem + "_hat.png")
        cv2.imwrite(str(hat_out), hat_bgr_out)
        cv2.imwrite(str(hat_out.parent / (hat_out.stem + "_preview.png")),
                    cv2.resize(hat_bgr_out, (512, 512), interpolation=cv2.INTER_NEAREST))
        print(f"[INFO] Latest hat copy → {hat_out}")

    # Clean up checkpoint now that training is complete
    if ckpt_path.exists():
        ckpt_path.unlink()
        print(f"[INFO] Checkpoint removed (training complete)")


if __name__ == "__main__":
    main()
