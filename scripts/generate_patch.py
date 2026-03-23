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
DEFAULT_HOSTS          = str(PROJECT_ROOT / "data" / "TRAINING LEG IMAGES _preprocessed")
DEFAULT_PRINTABLE_COLS = str(PROJECT_ROOT / "data" / "printable_colors.txt")
PERSON_CLASS    = 0
PERSON_COL_IDX  = 4 + PERSON_CLASS   # channel index in (4+nc, N_anchors) layout
TOP_K           = 50
IMG_SIZE        = 640
SUPPORTED_EXTS  = {".jpg", ".jpeg", ".png", ".bmp"}

# EOT hyper-parameters applied to the composite during training
EOT_SCALE_RANGE  = (0.50, 1.20)   # zoom range — simulates different viewing distances
EOT_ROT_RANGE    = 20.0           # degrees — wider rotation for head/body tilt
EOT_BLUR_MAX     = 5              # max Gaussian kernel size
EOT_BRIGHTNESS   = 0.30          # ± fraction — matches iter13; severe swings hurt convergence
EOT_PERSP_JITTER = 0.25          # max fractional corner displacement for perspective warp
EOT_JPEG_QUALITY = (55, 95)      # random JPEG quality range (simulates camera compression)
EOT_COLOR_JITTER = 0.20          # per-channel ± brightness shift (simulates ink colour shift)
EOT_PRINT_NOISE  = 8.0           # max std-dev of Gaussian noise added (simulates print grain)
EOT_HSV_HUE      = 12            # ± hue shift in degrees (matches iter13)
EOT_HSV_SAT      = 0.25          # ± saturation multiplier (matches iter13)
EOT_SHADOW_PROB  = 0.40          # probability of adding a random shadow strip (matches iter13)
EOT_GAMMA_RANGE  = (0.6, 1.8)    # narrower gamma range — avoids destroying patch signal


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adversarial patch generator (PGD v2) for YOLOv8n")
    p.add_argument("--lambda-attn", type=float, default=1.0,
                   help="Weight for attention redirection loss (default 1.0)")
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
    p.add_argument("--lr-min",     type=float, default=0.003,
                   help="Minimum LR for cosine decay schedule (default 0.003). "
                        "Set equal to --lr to disable decay.")
    p.add_argument("--eps",        type=float, default=1.0,
                   help="L-inf budget per pixel in [0,1] (default 1.0 = unconstrained)")
    p.add_argument("--topk",       type=int, default=TOP_K,
                   help="Top-k anchors used in loss (default 50)")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Images averaged per PGD step for smoother gradients (default 8)")
    p.add_argument("--no-eot",     action="store_true",
                   help="Disable EOT augmentation during training")
    p.add_argument("--geo-warmup", type=float, default=0.25,
                   help="Fraction of total steps with NO geometric EOT (photometric only)."
                        " Lets patch learn strong signal before MPS-CPU fallback kicks in."
                        " E.g. 0.25 = first 25%% of steps are photometric-only (default).")
    p.add_argument("--geo-ramp",   type=float, default=0.40,
                   help="Fraction of steps AFTER geo-warmup over which geo_prob ramps from 0 → 1"
                        " (default 0.40 = next 40%% of total steps). After warmup+ramp, full"
                        " geometric augmentation is applied every step.")
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
    p.add_argument("--checkpoint-every", type=int, default=300,
                   help="Save a resume checkpoint every N steps (default 300)")
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
                   help="When --bbox-placement is set: patch size as a fraction of the person"
                        " bbox height (default) or width (when --torso-width is set)."
                        " Default 0.25 ≈ 25%% of person height.")
    p.add_argument("--torso-width", action="store_true",
                   help="Size the bbox-guided patch from bbox WIDTH rather than height, so the"
                        " patch spans the full torso. patch_fraction is applied to bbox_width"
                        " (default 1.0 = full torso width; use --patch-fraction to scale).")
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

def load_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"[ERROR] Cannot read image: {path}")
    return img


def preprocess(bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 → float32 (1,3,H,W) [0,1]."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)


def load_host_pool(hosts_dir: str, single_host: str | None) -> list[np.ndarray]:
    """Return a list of preprocessed host BGR images at IMG_SIZE resolution."""
    if single_host:
        return [cv2.resize(load_bgr(single_host), (IMG_SIZE, IMG_SIZE))]
    d = Path(hosts_dir)
    if not d.exists():
        print(f"[WARN] --hosts-dir not found: {d}. Using blank canvas.")
        return [np.full((IMG_SIZE, IMG_SIZE, 3), 114, dtype=np.uint8)]
    imgs = [
        cv2.resize(load_bgr(str(p)), (IMG_SIZE, IMG_SIZE))
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
    out = img_t.clone()  # Use index_put for in-place safety with autograd
    out[:, :, row:row + ph, col:col + pw] = patch_t  # Composite the patch onto the image
    return out  # Return the modified image tensor


def apply_patch_resized(
    img_t: torch.Tensor,    # (1,3,H,W) float32 [0,1]
    patch_t: torch.Tensor,  # (1,3,ph,pw) float32 [0,1]  — must keep grad
    row: int,
    col: int,
    composite_height: int,    # height of the region to composite
    composite_width: int,     # width of the region to composite
) -> torch.Tensor:
    """
    Differentiably resize patch_t to composite_height×composite_width, then
    composite it onto img_t at (row, col). Gradients flow back through
    F.interpolate so the optimiser still updates the canonical patch.
    """
    # Always resize patch_t to (composite_height, composite_width) for assignment
    # Ensure patch_t is (1, 3, H, W)
    if patch_t.dim() == 3:
        patch_t = patch_t.unsqueeze(0)
    elif patch_t.dim() == 4 and patch_t.shape[0] != 1:
        raise ValueError(f"patch_t must have batch size 1, got {patch_t.shape}")
    # Interpolate to (1, 3, composite_height, composite_width)
    patch_scaled = torch.nn.functional.interpolate(
        patch_t,
        size=(composite_height, composite_width),
        mode="bilinear",
        align_corners=False
    )
    patch_assign = patch_scaled[0]  # (3, H, W)
    # Clamp patch placement and region to fit within image bounds
    img_shape = img_t.shape  # (1, 3, H, W)
    H, W = img_shape[2], img_shape[3]
    # Clamp row/col to be within image
    row = max(0, min(row, H - 1))
    col = max(0, min(col, W - 1))
    # Clamp composite_height/width so patch fits
    composite_height = min(composite_height, H - row)
    composite_width = min(composite_width, W - col)
    # Resize patch to new (possibly clamped) size
    patch_scaled = torch.nn.functional.interpolate(
        patch_t,
        size=(composite_height, composite_width),
        mode="bilinear",
        align_corners=False
    )
    patch_assign = patch_scaled[0]  # (3, H, W)
    # Debug print for assignment shapes
    if patch_assign.shape[1] != composite_height or patch_assign.shape[2] != composite_width:
        print(f"[DEBUG] patch_assign shape: {patch_assign.shape}, target: ({composite_height}, {composite_width})")
    if patch_assign.shape != (3, composite_height, composite_width):
        print(f"[ERROR] Patch assign shape mismatch: {patch_assign.shape} vs (3, {composite_height}, {composite_width})")
        raise ValueError(f"Patch assign shape mismatch: {patch_assign.shape} vs (3, {composite_height}, {composite_width})")
    out = img_t.clone()
    out[:, :, row:row + composite_height, col:col + composite_width] = patch_assign
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

    # Ensure output is always the same size as the original host image
    orig_h, orig_w = composite.shape[2], composite.shape[3]
    aug_h, aug_w = arr.shape[:2]
    # If augmented image is smaller, pad; if larger, center-crop
    if aug_h < orig_h or aug_w < orig_w:
        pad_h = max(0, orig_h - aug_h)
        pad_w = max(0, orig_w - aug_w)
        arr = cv2.copyMakeBorder(arr, pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2, cv2.BORDER_REPLICATE)
        aug_h, aug_w = arr.shape[:2]
    # Center-crop to original size
    start_y = (aug_h - orig_h) // 2
    start_x = (aug_w - orig_w) // 2
    arr = arr[start_y:start_y + orig_h, start_x:start_x + orig_w]

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


def eot_augment_differentiable(composite: torch.Tensor, geo_prob: float = 1.0) -> torch.Tensor:
    """
    Fully differentiable EOT augmentation with curriculum support.

    geo_prob : probability [0, 1] of applying geometric transforms (scale,
               rotation, perspective).  Set to 0.0 early in training to avoid
               the MPS grid_sample CPU fallback and let the patch learn a strong
               photometric-robust signal first.

    All geometric transforms are applied via torch.nn.functional.grid_sample so
    gradients flow back to patch pixels.  Photometric transforms are pure tensor
    ops.  JPEG uses a straight-through estimator.
    """
    t = composite                          # (1, C, H, W) float32 [0, 1]
    if t.shape[1] == 1:
        # Repeat single channel to 3 channels
        t = t.repeat(1, 3, 1, 1)
    elif t.shape[1] == 4:
        # Drop alpha channel if present
        t = t[:, :3, :, :]
    elif t.shape[1] != 3:
        raise ValueError(f"Input to eot_augment_differentiable must have 1 or 3 channels, got {t.shape[1]}")
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
    # 5. Combined affine (scale/rotation) and perspective as a single grid_sample
    if random.random() < geo_prob:
        sc        = random.uniform(*EOT_SCALE_RANGE)
        angle_rad = math.radians(random.uniform(-EOT_ROT_RANGE, EOT_ROT_RANGE))
        ca = math.cos(angle_rad) / sc
        sa = math.sin(angle_rad) / sc
        # Affine matrix (3x3)
        affine = np.array([
            [ca, -sa, 0.0],
            [sa,  ca, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        # Perspective jitter
        jx = EOT_PERSP_JITTER * W
        jy = EOT_PERSP_JITTER * H
        src_pts = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
        dst_pts = np.float32([
            [random.uniform(0, jx),     random.uniform(0, jy)],
            [random.uniform(W - jx, W), random.uniform(0, jy)],
            [random.uniform(W - jx, W), random.uniform(H - jy, H)],
            [random.uniform(0, jx),     random.uniform(H - jy, H)],
        ])
        persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
        # Compose affine and perspective (persp @ affine)
        combined = np.matmul(persp, affine)
        # Invert: grid_sample needs input coords for each output pixel
        H_inv = np.linalg.inv(combined)
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
        combined_grid = torch.stack([gx, gy], dim=-1).reshape(1, H, W, 2)
        t = torch.nn.functional.grid_sample(
            t, combined_grid, mode='bilinear', padding_mode='zeros', align_corners=False
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

    # 9. JPEG simulation (pure-tensor blocky quantization approximation)
    #    Simulates block artifacts and quantization without CPU/GPU transfer.
    block_size = random.choice([8, 16])
    quant_levels = random.randint(8, 32)
    t_blocky = t.clone()
    _, _, H, W = t_blocky.shape
    for y in range(0, H, block_size):
        for x in range(0, W, block_size):
            block = t_blocky[:, :, y:y+block_size, x:x+block_size]
            block_mean = block.mean(dim=[2,3], keepdim=True)
            block_quant = torch.round(block_mean * quant_levels) / quant_levels
            t_blocky[:, :, y:y+block_size, x:x+block_size] = block_quant
    t = t_blocky.detach() + (t - t.detach())  # STE: forward=blocky, backward=clean

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
    if isinstance(patch_size, tuple):
        patch_height, patch_width = patch_size
    else:
        patch_height = patch_size
        patch_width = int(patch_size * 0.25)
    canvas = np.zeros((patch_height, patch_width), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_DUPLEX
    scale, thickness = 1.0, 2
    for s_int in range(5, 400):
        s = s_int * 0.1
        th = max(2, int(s * 3))
        (_, ch), _ = cv2.getTextSize(letter, font, s, thickness=th)
        if ch >= patch_height * 0.72:
            scale, thickness = s, th
            break
    (tw, th), _ = cv2.getTextSize(letter, font, scale, thickness)
    x = max(0, (patch_width - tw) // 2)
    y = min(patch_height - 4, (patch_height + th) // 2)
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
    Return a (1, 3, N, N) square float32 tensor in [0, 1].

    The patch is always square. The 1:4 leg shape is enforced at compositing
    time via a per-frame rotated mask (see composite_leg_patch). Keeping the
    tensor square means PyTorch grid_sample / affine ops work without padding.

    Modes: uniform, gaussian, checkerboard, stripes, salt_pepper, gray
    """
    N = patch_size if isinstance(patch_size, int) else max(patch_size)

    if mode == "gaussian":
        t = torch.randn(1, 3, N, N) * 0.2 + 0.5
        t = t.clamp(0.0, 1.0)
    elif mode == "checkerboard":
        cell = 8
        arr = np.zeros((N, N), dtype=np.float32)
        for r in range(N):
            for c in range(N):
                if ((r // cell) + (c // cell)) % 2 == 0:
                    arr[r, c] = 1.0
        t = torch.from_numpy(np.stack([arr, arr, arr], axis=0)).unsqueeze(0)
    elif mode == "stripes":
        band = 16
        arr = np.zeros((N, N), dtype=np.float32)
        for c in range(N):
            if (c // band) % 2 == 0:
                arr[:, c] = 1.0
        t = torch.from_numpy(np.stack([arr, arr, arr], axis=0)).unsqueeze(0)
    elif mode == "salt_pepper":
        arr = np.full((N, N), 0.5, dtype=np.float32)
        mask = np.random.rand(N, N)
        arr[mask < 0.15] = 0.0
        arr[mask > 0.85] = 1.0
        t = torch.from_numpy(np.stack([arr, arr, arr], axis=0)).unsqueeze(0)
    elif mode == "gray":
        t = torch.full((1, 3, N, N), 0.447)
    else:  # uniform (default)
        t = torch.rand(1, 3, N, N)

    return t.float().to(device)


def make_leg_mask(
    hip: np.ndarray,
    ankle: np.ndarray,
    img_h: int,
    img_w: int,
    width_frac: float = 0.25,
) -> np.ndarray:
    """
    Return a float32 (img_h, img_w) binary mask with a filled rotated rectangle
    aligned between hip and ankle.  width_frac controls how wide the rectangle
    is relative to the leg length (0.25 → 1:4 aspect ratio).

    The mask is used both to composite the patch onto the host image and to
    restrict gradients to the leg region during training.
    """
    leg_vec = ankle - hip
    leg_length = float(np.linalg.norm(leg_vec))
    if leg_length < 4:
        return np.zeros((img_h, img_w), dtype=np.float32)
    rect_h = int(leg_length)
    rect_w = max(4, int(leg_length * width_frac))
    center = ((hip + ankle) / 2).tolist()
    angle = math.degrees(math.atan2(leg_vec[1], leg_vec[0]))
    box = cv2.boxPoints((tuple(center), (rect_h, rect_w), angle))
    box = box.astype(np.int32)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.drawContours(mask, [box], 0, 1, -1)
    return mask.astype(np.float32)


def composite_leg_patch(
    host_t: torch.Tensor,        # (1, 3, H, W) float32 [0,1]
    patch: torch.Tensor,         # (1, 3, N, N) float32 [0,1]  requires_grad
    hip: np.ndarray,             # (2,) float32 pixel coords
    ankle: np.ndarray,           # (2,) float32 pixel coords
    img_h: int,
    img_w: int,
    width_frac: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Composite the square patch onto host_t inside the rotated 1:4 leg rectangle.

    Strategy:
      1. Build a float mask covering the rotated leg rectangle (numpy, no grad).
      2. Resize the square patch to (rect_h, rect_w) — gradients flow through
         F.interpolate back to patch pixels.
      3. Place the resized patch into a zero canvas the same size as the host.
      4. Blend:  out = host * (1 - mask) + patch_canvas * mask

    Returns (composite, mask_t) where mask_t is (1,1,H,W) for loss weighting.
    """
    leg_vec = ankle - hip
    leg_length = float(np.linalg.norm(leg_vec))
    rect_h = max(4, int(leg_length))
    rect_w = max(4, int(leg_length * width_frac))

    # --- differentiable patch resize -----------------------------------------
    patch_resized = torch.nn.functional.interpolate(
        patch, size=(rect_h, rect_w), mode="bilinear", align_corners=False
    )  # (1, 3, rect_h, rect_w)

    # --- build mask and placement canvas (numpy, no grad) --------------------
    mask_np = make_leg_mask(hip, ankle, img_h, img_w, width_frac)
    mask_t = torch.from_numpy(mask_np).to(host_t.device).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    # Axis-aligned bounding box of the rotated rectangle — used to place the
    # resized patch onto the canvas before masking clips it to the true shape.
    center = (hip + ankle) / 2
    row = int(center[1]) - rect_h // 2
    col = int(center[0]) - rect_w // 2
    row = max(0, min(row, img_h - rect_h))
    col = max(0, min(col, img_w - rect_w))

    # Canvas: start from host, paint patch into the AABB; mask clips the shape.
    canvas = host_t.clone()
    ph = min(rect_h, img_h - row)
    pw = min(rect_w, img_w - col)
    canvas[:, :, row:row + ph, col:col + pw] = patch_resized[:, :, :ph, :pw]

    composite = host_t * (1.0 - mask_t) + canvas * mask_t
    return composite, mask_t


# ---------------------------------------------------------------------------
# Eval sample visualisation
# ---------------------------------------------------------------------------

def save_eval_samples(
    host_pool_bgr: list,
    patch_t: torch.Tensor,
    pose_keypoints: list,
    out_dir: Path,
    yolo_wrapper,
    n_samples: int = 5,
    width_frac: float = 0.25,
) -> None:
    """
    Save n_samples annotated images showing:
      - BLUE box + confidence : highest-confidence pre-patch detection
      - CYAN outline          : leg rectangle where the patch was composited
      - GREEN box + confidence: most confident post-patch detection
      - "SUPPRESSED"          : if no person detected post-patch
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(host_pool_bgr)
    idxs = [int(round(i * (n - 1) / max(n_samples - 1, 1))) for i in range(n_samples)]
    idxs = list(dict.fromkeys(idxs))[:n_samples]
    font = cv2.FONT_HERSHEY_SIMPLEX

    for rank, idx in enumerate(idxs):
        orig_bgr = host_pool_bgr[idx]          # never modify the original
        img_h, img_w = orig_bgr.shape[:2]

        # --- Pre-patch detection on clean host (BLUE) --------------------
        with torch.no_grad():
            pre_results = yolo_wrapper.predict(
                source=orig_bgr, conf=0.01, classes=[PERSON_CLASS], verbose=False
            )
        pre_boxes = pre_results[0].boxes
        pre_ann = None   # (x1,y1,x2,y2, conf) of the best pre-patch box
        if pre_boxes is not None and len(pre_boxes) > 0:
            best_j = int(pre_boxes.conf.cpu().argmax())
            pre_ann = (*pre_boxes.xyxy.cpu()[best_j].numpy().astype(int),
                       float(pre_boxes.conf.cpu()[best_j]))

        # --- Composite patch onto the CLEAN host -------------------------
        kpts_list = pose_keypoints[idx] if idx < len(pose_keypoints) else []
        hip_px = ankle_px = None
        if len(kpts_list) > 0:
            person_kpts = kpts_list[0]
            x_hip, y_hip, c_hip = person_kpts[12]
            x_ankle, y_ankle, c_ankle = person_kpts[16]
            if c_hip >= 0.3 and c_ankle >= 0.3:
                hip_px = np.array([x_hip, y_hip], dtype=np.float32)
                ankle_px = np.array([x_ankle, y_ankle], dtype=np.float32)

        if hip_px is not None:
            host_t = preprocess(orig_bgr)
            with torch.no_grad():
                comp_t, _ = composite_leg_patch(
                    host_t, patch_t.cpu(), hip_px, ankle_px, img_h, img_w, width_frac
                )
            # img now contains the actual patch pixels blended onto the host
            img = cv2.cvtColor(
                (comp_t.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                cv2.COLOR_RGB2BGR
            )
        else:
            img = orig_bgr.copy()

        # --- Annotate: pre-patch detection (BLUE) ------------------------
        if pre_ann is not None:
            bx1, by1, bx2, by2, bconf = pre_ann
            cv2.rectangle(img, (bx1, by1), (bx2, by2), (220, 100, 0), 2)
            cv2.putText(img, f"pre {bconf:.3f}", (bx1 + 4, by1 - 6),
                        font, 0.55, (220, 100, 0), 1, cv2.LINE_AA)

        # --- Post-patch detection (GREEN, most confident only) -----------
        with torch.no_grad():
            post_results = yolo_wrapper.predict(
                source=img, conf=0.01, classes=[PERSON_CLASS], verbose=False
            )
        post_boxes = post_results[0].boxes
        if post_boxes is not None and len(post_boxes) > 0:
            best_k = int(post_boxes.conf.cpu().argmax())
            post_conf = float(post_boxes.conf.cpu()[best_k])
            px1, py1, px2, py2 = post_boxes.xyxy.cpu()[best_k].numpy().astype(int)
            cv2.rectangle(img, (px1, py1), (px2, py2), (0, 200, 0), 2)
            cv2.putText(img, f"post {post_conf:.3f}", (px1 + 4, py2 + 16),
                        font, 0.55, (0, 200, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "SUPPRESSED", (img_w // 2 - 80, img_h // 2),
                        font, 1.0, (0, 200, 0), 2, cv2.LINE_AA)

        # --- Legend -------------------------------------------------------
        cv2.putText(img, "BLUE=pre-patch  GREEN=post-patch",
                    (6, img_h - 8), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imwrite(str(out_dir / f"eval_{rank + 1:02d}.jpg"), img)

    print(f"[INFO] Eval samples → {out_dir}  ({len(idxs)} images)")


def _compute_saliency(tensor_img: torch.Tensor, torch_model: torch.nn.Module) -> np.ndarray:
    """
    Compute saliency map for a (1,3,H,W) float32 image tensor.
    Returns a float32 (H,W) array in [0,1], representing
    max-channel |d(mean person score)/d(pixel)|, normalised.
    """
    inp = tensor_img.detach().requires_grad_(True)
    with torch.enable_grad():
        pred = torch_model(inp)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        score = pred[0, PERSON_COL_IDX, :].mean()
        score.backward()
    sal = inp.grad.abs().max(dim=1)[0].squeeze(0).cpu().numpy()  # (H,W)
    sal_min, sal_max = sal.min(), sal.max()
    if sal_max > sal_min:
        sal = (sal - sal_min) / (sal_max - sal_min)
    return sal.astype(np.float32)


def _sal_overlay(bgr_img: np.ndarray, sal: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Blend a [0,1] saliency map onto a BGR image using the JET colormap."""
    heat = cv2.applyColorMap((sal * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.resize(heat, (bgr_img.shape[1], bgr_img.shape[0]))
    return cv2.addWeighted(bgr_img, 1.0 - alpha, heat, alpha, 0)


def save_saliency_maps(
    host_pool_bgr: list,
    patch_t: torch.Tensor,
    pose_keypoints: list,
    torch_model: torch.nn.Module,
    out_dir: Path,
    n_samples: int = 3,
    width_frac: float = 0.25,
) -> None:
    """
    For n_samples random hosts save side-by-side images:
      left  — clean host with pre-patch saliency heatmap
      right — patched host with post-patch saliency heatmap
    Helps diagnose whether the patch is redirecting detector attention
    from the upper body down to the leg region.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(host_pool_bgr)
    idxs = random.sample(range(n), min(n_samples, n))
    font = cv2.FONT_HERSHEY_SIMPLEX

    torch_model.eval()
    for p in torch_model.parameters():
        p.requires_grad_(False)

    for rank, idx in enumerate(idxs):
        host_bgr = host_pool_bgr[idx]
        img_h, img_w = host_bgr.shape[:2]

        host_t = preprocess(host_bgr)  # (1,3,H,W)

        # --- Pre-patch saliency ------------------------------------------
        sal_pre = _compute_saliency(host_t, torch_model)
        left = _sal_overlay(host_bgr.copy(), sal_pre)
        cv2.putText(left, "PRE-PATCH SALIENCY", (6, 22), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(left, "PRE-PATCH SALIENCY", (6, 22), font, 0.65, (0, 0, 0), 1, cv2.LINE_AA)

        # --- Post-patch saliency -----------------------------------------
        kpts_list = pose_keypoints[idx] if idx < len(pose_keypoints) else []
        hip_px = ankle_px = None
        if len(kpts_list) > 0:
            pk = kpts_list[0]
            x_hip, y_hip, c_hip = pk[12]
            x_ankle, y_ankle, c_ankle = pk[16]
            if c_hip >= 0.3 and c_ankle >= 0.3:
                hip_px = np.array([x_hip, y_hip], dtype=np.float32)
                ankle_px = np.array([x_ankle, y_ankle], dtype=np.float32)

        if hip_px is not None:
            with torch.no_grad():
                comp_t, _ = composite_leg_patch(
                    host_t, patch_t.cpu(), hip_px, ankle_px, img_h, img_w, width_frac
                )
            sal_post = _compute_saliency(comp_t, torch_model)
            comp_bgr = cv2.cvtColor(
                (comp_t.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                cv2.COLOR_RGB2BGR
            )
            # Draw leg outline on right panel
            mask_np = make_leg_mask(hip_px, ankle_px, img_h, img_w, width_frac)
            contours, _ = cv2.findContours(
                mask_np.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(comp_bgr, contours, -1, (0, 255, 255), 2)
        else:
            sal_post = _compute_saliency(host_t, torch_model)
            comp_bgr = host_bgr.copy()

        right = _sal_overlay(comp_bgr, sal_post)
        cv2.putText(right, "POST-PATCH SALIENCY", (6, 22), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(right, "POST-PATCH SALIENCY", (6, 22), font, 0.65, (0, 0, 0), 1, cv2.LINE_AA)

        # --- Side-by-side ------------------------------------------------
        divider = np.full((img_h, 4, 3), 50, dtype=np.uint8)
        combined = np.concatenate([left, divider, right], axis=1)
        cv2.imwrite(str(out_dir / f"saliency_{rank + 1:02d}.jpg"), combined)

    print(f"[INFO] Saliency maps → {out_dir}  ({len(idxs)} images)")


# ---------------------------------------------------------------------------
# (save_eval_samples docstring was here — now a proper function above)
# ---------------------------------------------------------------------------

# Stub kept so old call-sites that pass extra kwargs don't crash at import time.
# The real logic is in save_eval_samples above.
def _save_eval_samples_legacy(**_kwargs) -> None:
    pass


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
    torso_width: bool = False,
):
    # TODO: Reimplement the actual logic. For now, return dummy values to avoid errors.

    return 0.0, []

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    # LEG PATCH PIPELINE ONLY
    # Remove all non-leg-patch logic and variables
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
    print(f"[INFO] Steps      : {args.steps}  |  LR: {args.lr} → {args.lr_min} (cosine)  |  ε: {args.eps}")
    print(f"[INFO] Batch size : {args.batch_size} images/step")
    print(f"[INFO] Loss mode  : mean (confidence suppression)")
    # Load model
    yolo = YOLO(args.model)
    torch_model: torch.nn.Module = yolo.model
    torch_model.eval().to(device)
    for p in torch_model.parameters():
        p.requires_grad_(False)
    # Load host images
    host_pool_bgr = load_host_pool(args.hosts_dir, args.host)
    host_pool_t   = [preprocess(h).to(device) for h in host_pool_bgr]
    # Hard-example mining: per-host EMA confidence tracker
    host_conf_ema = np.ones(len(host_pool_t), dtype=np.float32)
    # Load printable colours for NPS loss
    printable_colors: torch.Tensor | None = None
    if args.alpha > 0 and Path(args.printable_colors).exists():
        printable_colors = load_printable_colors(args.printable_colors, device)
        print(f"[INFO] Printable colours loaded: {printable_colors.shape[0]} swatches")
    elif args.alpha > 0:
        print(f"[WARN] --alpha {args.alpha} set but printable_colors file not found; NPS disabled")
    # 4. Initialise patch (or resume from checkpoint)
    ckpt_path  = out_path.parent / (out_path.stem + "_ckpt.pt")
    start_step = 0
    best_loss  = float("inf")
    patch = init_patch(args.init, patch_size, device)
    patch.requires_grad_(True)
    best_patch = patch.detach().clone()
    # No hat patch, no style/letter/printable colors, no bbox/torso logic

    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        patch      = ckpt["patch"].to(device).requires_grad_(True)
        start_step = ckpt["step"]
        best_loss  = ckpt["best_loss"]
        best_patch = ckpt["best_patch"].to(device)
        print(f"[INFO] Resumed from checkpoint  step={start_step}  best_loss={best_loss:.6f}")

    # Hat/crown patch — separate tensor, same size as torso patch.
    # Stored and updated independently so gradients are clean.
    hat_patch: torch.Tensor | None = None
    best_hat_patch: torch.Tensor | None = None
    # Hat patch forcibly disabled; do not allocate

    # ------------------------------------------------------------------
    # 5. Baseline: average max-confidence person detection, all hosts, no patch
    # ------------------------------------------------------------------
    print(f"[INFO] Computing clean baseline confidence ({len(host_pool_bgr)} hosts, no patch) …")
    clean_baseline = 0.0
    baseline_anchor_boxes = []
    print(f"[INFO] Clean baseline mean confidence : {clean_baseline:.6f}")

    # ------------------------------------------------------------------
    # 5b. Pre-extract pose keypoints for all host images (once, before loop)
    # ------------------------------------------------------------------
    print("[INFO] Extracting pose keypoints from host images …")
    pose_model = YOLO("yolov8n-pose.pt")
    pose_keypoints: list = []
    for h_bgr in host_pool_bgr:
        results = pose_model(h_bgr, verbose=False)
        kpts = results[0].keypoints.data.cpu().numpy() if results[0].keypoints is not None else []
        pose_keypoints.append(kpts)
    print(f"[INFO] Pose extraction done  ({len(pose_keypoints)} images)")

    # ------------------------------------------------------------------
    # 6. PGD loop with EOT augmentation + multi-host + leg-patch placement
    # ------------------------------------------------------------------
    for step in range(start_step + 1, args.steps + 1):
        if patch.grad is not None:
            patch.grad.zero_()

        obj_losses = []
        attn_losses = []
        prog = (step - 1) / max(args.steps - 1, 1)   # 0.0 → 1.0

        # Cosine LR decay
        lr_curr = args.lr_min + 0.5 * (args.lr - args.lr_min) * (1.0 + math.cos(math.pi * prog))

        # EOT geo curriculum probability
        if not args.no_eot:
            w = args.geo_warmup
            r = args.geo_ramp
            if prog < w:
                geo_prob = 0.0
            elif prog < w + r:
                geo_prob = (prog - w) / r
            else:
                geo_prob = 1.0
        else:
            geo_prob = 1.0

        for _ in range(args.batch_size):
            # (a) Sample host — hard-mining weighted or uniform
            if args.hard_mining and len(host_pool_t) > 1:
                samp_w = _softmax_weights(host_conf_ema, args.hard_temp)
                idx = random.choices(range(len(host_pool_t)), weights=samp_w, k=1)[0]
            else:
                idx = random.randrange(len(host_pool_t))

            # (b) Get pre-extracted pose keypoints for this host
            kpts_arr = pose_keypoints[idx]
            if len(kpts_arr) == 0:
                continue
            person_kpts = kpts_arr[0]  # first detected person
            x_hip,    y_hip,    conf_hip    = person_kpts[12]
            x_ankle,  y_ankle,  conf_ankle  = person_kpts[16]
            if conf_hip < 0.3 or conf_ankle < 0.3:
                continue

            hip_px    = np.array([float(x_hip),   float(y_hip)],   dtype=np.float32)
            ankle_px  = np.array([float(x_ankle), float(y_ankle)], dtype=np.float32)

            # (c) Differentiable composite: square patch → rotated 1:4 rectangle on host
            host_t = host_pool_t[idx]
            composite, mask_t = composite_leg_patch(
                host_t, patch, hip_px, ankle_px,
                IMG_SIZE, IMG_SIZE, width_frac=0.25
            )

            # (d) EOT augmentation
            if not args.no_eot:
                composite_aug = eot_augment_differentiable(composite, geo_prob=geo_prob)
            else:
                composite_aug = composite

            if composite_aug.shape[-2:] != (IMG_SIZE, IMG_SIZE):
                composite_aug = torch.nn.functional.interpolate(
                    composite_aug, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
                )

            # (e) Forward loss
            sample_loss = forward_person_loss(
                torch_model, composite_aug, args.topk, device,
                bbox=None, anchor_centers=None, iou_sigma=args.iou_sigma
            )
            obj_losses.append(sample_loss)

            # Update hard-mining EMA
            if args.hard_mining:
                host_conf_ema[idx] = (
                    0.9 * host_conf_ema[idx] + 0.1 * sample_loss.detach().item()
                )

            # (f) Saliency / attention redirection:
            #     Compute saliency weights from a DETACHED copy of the image so
            #     the saliency map is a constant spatial weighting.  Then apply
            #     those weights to the ATTACHED composite_aug so gradients flow
            #     back to patch pixels correctly.
            if args.lambda_attn > 0:
                # Step 1: get saliency map as a constant spatial weight (no grad to patch)
                ca = composite_aug.detach().requires_grad_(True)
                with torch.enable_grad():
                    pred_sal = torch_model(ca)
                    if isinstance(pred_sal, (list, tuple)):
                        pred_sal = pred_sal[0]
                    pred_sal[0, PERSON_COL_IDX, :].mean().backward()
                with torch.no_grad():
                    sal_weight = ca.grad.abs().max(dim=1, keepdim=True)[0]  # (1,1,H,W)
                    sal_weight = sal_weight / (sal_weight.sum() + 1e-8)     # normalise to sum-1

                # Step 2: resize mask to augmented-image resolution
                mask_aug = torch.nn.functional.interpolate(
                    mask_t, size=composite_aug.shape[-2:], mode="nearest"
                )  # (1,1,H,W)

                # Step 3: weighted pixel mean of the ATTACHED composite_aug
                #   L_in:  saliency-weighted mean of patch pixels  (want HIGH → push toward 1)
                #   L_out: saliency-weighted mean of non-patch pixels (want LOW → push toward 0)
                #   Negated because PGD does gradient descent
                eps_mask = 1e-8
                L_in  = (sal_weight * mask_aug * composite_aug).sum() \
                        / mask_aug.sum().clamp(min=eps_mask)
                L_out = (sal_weight * (1.0 - mask_aug) * composite_aug).sum() \
                        / (1.0 - mask_aug).sum().clamp(min=eps_mask)
                attn_loss = -(L_in - args.lambda_attn * L_out)
                attn_losses.append(attn_loss)

        if not obj_losses:
            continue  # no valid samples this step (all hosts lacked confident keypoints)

        # (f) Mean loss + regularisation
        loss = sum(obj_losses) / len(obj_losses)
        if attn_losses:
            loss = loss + sum(attn_losses) / len(attn_losses)
        if args.alpha > 0 and printable_colors is not None:
            loss = loss + args.alpha * nps_loss(patch, printable_colors)
        if args.beta > 0:
            loss = loss + args.beta * tv_loss(patch)

        # (g) Backward + PGD step
        loss.backward()
        with torch.no_grad():
            patch.data -= lr_curr * patch.grad.sign()
            patch.data.clamp_(0.0, 1.0)
        patch.grad = None

        if loss.item() < best_loss:
            best_loss  = loss.item()
            best_patch = patch.detach().clone()

        # Periodic checkpoint
        if step % args.checkpoint_every == 0:
            torch.save({
                "step":       step,
                "patch":      patch.detach().cpu(),
                "best_patch": best_patch.cpu(),
                "best_loss":  best_loss,
            }, ckpt_path)

        if args.verbose and step % 10 == 0:
            attn_str = f"  attn={sum(l.item() for l in attn_losses)/len(attn_losses):.4f}" if attn_losses else ""
            print(f"  Step {step:>5d}/{args.steps}  loss={loss.item():.6f}  best={best_loss:.6f}  lr={lr_curr:.5f}  geo={geo_prob:.2f}{attn_str}")

    # ------------------------------------------------------------------
    # 6b. Post-training summary
    # ------------------------------------------------------------------
    print(f"[INFO] Training complete. Best loss: {best_loss:.6f}")
    clean_final = 0.0
    reduction = 0.0
    print(f"[INFO] Confidence reduction : {clean_baseline:.6f} → {clean_final:.6f}  ({reduction:.1f}% suppression)")

    # ------------------------------------------------------------------
    # 7. Save patch + preview
    # ------------------------------------------------------------------
    patch_np  = best_patch.squeeze(0).permute(1, 2, 0).cpu().numpy()
    patch_bgr = cv2.cvtColor((patch_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # --- 7a. Auto-versioned iteration folder ----------------------------
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

    # --- 7b. Eval sample images (5) — pre/post detection boxes ----------
    save_eval_samples(
        host_pool_bgr=host_pool_bgr,
        patch_t=best_patch,
        pose_keypoints=pose_keypoints,
        out_dir=iter_dir / "eval_samples",
        yolo_wrapper=yolo,
        n_samples=5,
    )

    # --- 7c. Saliency maps (3) — pre/post heatmaps side-by-side ----------
    save_saliency_maps(
        host_pool_bgr=host_pool_bgr,
        patch_t=best_patch,
        pose_keypoints=pose_keypoints,
        torch_model=torch_model,
        out_dir=iter_dir / "saliency_maps",
        n_samples=3,
    )

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
        f.write(f"lr             : {args.lr} → {args.lr_min} (cosine decay)\n")
        f.write(f"eps            : {args.eps}\n")
        f.write(f"init           : {args.init}\n")
        f.write(f"batch_size     : {args.batch_size}\n")
        f.write(f"topk           : {args.topk}\n")
        f.write(f"alpha_nps      : {args.alpha}\n")
        f.write(f"beta_tv        : {args.beta}\n")
        f.write(f"eot            : {not args.no_eot}\n")
        f.write(f"eot_geo_warmup : {args.geo_warmup:.0%} of steps (photometric only)\n")
        f.write(f"eot_geo_ramp   : {args.geo_ramp:.0%} of steps (geo_prob 0→1)\n")
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
        f.write(f"torso_width    : {args.torso_width}\n")
        f.write(f"hat_patch      : {args.hat_patch}\n")
        # Hat patch parameters removed
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
        pass  # v-shape and chin output removed

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
