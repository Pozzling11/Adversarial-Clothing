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
import hashlib
import json
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
SUPPORTED_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif"}

# EOT hyper-parameters applied to the composite during training
EOT_SCALE_RANGE       = (0.50, 1.20)  # close-up zoom band — realistic 2–8m viewing distance
EOT_SCALE_RANGE_MED   = (0.35, 0.55)  # medium-distance band (~8–10m)
EOT_SCALE_RANGE_FAR   = (0.15, 0.35)  # far-distance band (~10–15m)
EOT_MED_VIEW_PROB     = 0.25          # 25% of steps use medium-distance scale
EOT_FAR_VIEW_PROB     = 0.15          # 15% of steps use far-distance scale
                                       # 60% close / 25% medium / 15% far
                                       # Keeps close-up gradient dominant while building
                                       # far-distance robustness
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
EOT_WRINKLE_PROB = 0.50          # probability of applying cloth-wrinkle deformation per step
EOT_WRINKLE_STRENGTH = 0.08      # max displacement as fraction of patch size (0.08 = subtle folds)
EOT_POS_JITTER  = 0.10           # ±10% positional jitter relative to patch region size
                                   # Matches real t-shirt shift (~3-5cm on ~40cm torso)
                                   # and covers a full stride-16 anchor cell


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adversarial patch generator (PGD v2) for YOLOv8n")
    p.add_argument("--lambda-attn", type=float, default=0.0,
                   help="Weight for attention redirection loss (default 0.0 = disabled)")
    p.add_argument("--log-file", default=None,
                   help="Write all training output to this file in addition to stdout")
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
    p.add_argument("--lr",         type=float, default=0.01,
                   help="PGD step size (default 0.01)")
    p.add_argument("--lr-min",     type=float, default=0.001,
                   help="Minimum LR for cosine decay schedule (default 0.001). "
                        "Set equal to --lr to disable decay.")
    p.add_argument("--eps",        type=float, default=1.0,
                   help="L-inf budget per pixel in [0,1] (default 1.0 = unconstrained)")
    p.add_argument("--topk",       type=int, default=TOP_K,
                   help="Top-k anchors used in loss (default 50)")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Images averaged per PGD step for smoother gradients (default 16)")
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
    p.add_argument("--dual-patch", action="store_true",
                   help="Train a torso patch jointly with the leg patch. Both are composited "
                        "onto the host in a single forward pass and updated together via PGD.")
    p.add_argument("--torso-out", default=None,
                   help="Output PNG path for the torso patch (default: <out>_torso.png).")
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
# Training log helper
# ---------------------------------------------------------------------------

_log_fh = None  # file handle opened by main() if --log-file is set

def _log(msg: str) -> None:
    """Print to stdout and optionally write to the log file, flushing immediately."""
    print(msg, flush=True)
    if _log_fh is not None:
        _log_fh.write(msg + "\n")
        _log_fh.flush()


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
# Startup cache — avoids redundant YOLO / pose inference on unchanged hosts
# ---------------------------------------------------------------------------

CACHE_DIR = PROJECT_ROOT / ".cache"


def _host_cache_key(hosts_dir: str, single_host: str | None, model_path: str) -> str:
    """
    Compute a hex digest that changes when the host image set or model changes.
    Uses file paths + sizes + mtimes (fast, no full-content hashing).
    """
    h = hashlib.sha256()
    h.update(model_path.encode())
    h.update(str(Path(model_path).stat().st_size).encode())

    if single_host:
        p = Path(single_host)
        h.update(str(p).encode())
        h.update(f"{p.stat().st_size}:{p.stat().st_mtime_ns}".encode())
    else:
        d = Path(hosts_dir)
        if d.exists():
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in SUPPORTED_EXTS and not p.name.startswith("._"):
                    h.update(str(p).encode())
                    h.update(f"{p.stat().st_size}:{p.stat().st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def _load_startup_cache(cache_key: str) -> dict | None:
    """Load cached baseline detections + pose keypoints, or None if stale/missing."""
    cache_file = CACHE_DIR / f"startup_{cache_key}.pt"
    if not cache_file.exists():
        return None
    try:
        data = torch.load(cache_file, map_location="cpu", weights_only=False)
        if data.get("version") != 2:
            return None
        return data
    except Exception:
        return None


def _save_startup_cache(
    cache_key: str,
    baseline_confs: list[float],
    host_bboxes: list,
    pose_keypoints: list,
) -> None:
    """Persist baseline detections + pose keypoints to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"startup_{cache_key}.pt"
    torch.save({
        "version": 2,
        "baseline_confs": baseline_confs,
        "host_bboxes": host_bboxes,
        "pose_keypoints": pose_keypoints,
    }, cache_file)
    print(f"[CACHE] Saved startup cache → {cache_file}")


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

    # 4. Random scale (zoom) — trimodal: 60% close / 25% medium / 15% far
    # Each band uses log-uniform so sub-octaves get equal coverage.
    _r = random.random()
    if _r < EOT_FAR_VIEW_PROB:
        sc = math.exp(random.uniform(math.log(EOT_SCALE_RANGE_FAR[0]), math.log(EOT_SCALE_RANGE_FAR[1])))
    elif _r < EOT_FAR_VIEW_PROB + EOT_MED_VIEW_PROB:
        sc = math.exp(random.uniform(math.log(EOT_SCALE_RANGE_MED[0]), math.log(EOT_SCALE_RANGE_MED[1])))
    else:
        sc = math.exp(random.uniform(math.log(EOT_SCALE_RANGE[0]), math.log(EOT_SCALE_RANGE[1])))
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


def eot_augment_differentiable(composite: torch.Tensor, geo_prob: float = 1.0, allow_far_view: bool = True, allow_med_view: bool = True) -> torch.Tensor:
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

    # ── Geometric (MPS-native — no grid_sample) ────────────────────────────
    # 5. Scale + rotation + perspective using only F.interpolate, F.pad, and
    #    per-row/column shifts.  All ops have MPS backward support.
    if random.random() < geo_prob:
        # Trimodal scale: 60% close / 25% medium / 15% far (log-uniform per band)
        # Gated by shoulder width so small subjects aren't shrunk to sub-stride sizes:
        #   far  (0.15–0.35×) requires shoulders ≥80px → tile ≥12px after scaling
        #   med  (0.35–0.55×) requires shoulders ≥50px → tile ≥18px after scaling
        _r = random.random()
        if allow_far_view and _r < EOT_FAR_VIEW_PROB:
            sc = math.exp(random.uniform(math.log(EOT_SCALE_RANGE_FAR[0]), math.log(EOT_SCALE_RANGE_FAR[1])))
        elif allow_med_view and _r < EOT_FAR_VIEW_PROB + EOT_MED_VIEW_PROB:
            sc = math.exp(random.uniform(math.log(EOT_SCALE_RANGE_MED[0]), math.log(EOT_SCALE_RANGE_MED[1])))
        else:
            sc = math.exp(random.uniform(math.log(EOT_SCALE_RANGE[0]), math.log(EOT_SCALE_RANGE[1])))

        new_h = max(1, int(H * sc))
        new_w = max(1, int(W * sc))

        if sc >= 1.0:
            # Zoom in — resize up then centre-crop back to (H, W)
            t = torch.nn.functional.interpolate(t, size=(new_h, new_w), mode='bilinear', align_corners=False)
            y0 = (new_h - H) // 2
            x0 = (new_w - W) // 2
            t = t[:, :, y0:y0 + H, x0:x0 + W]
        else:
            # Zoom out — resize down then pad to (H, W)
            t = torch.nn.functional.interpolate(t, size=(new_h, new_w), mode='bilinear', align_corners=False)
            pad_top = (H - new_h) // 2
            pad_bot = H - new_h - pad_top
            pad_left = (W - new_w) // 2
            pad_right = W - new_w - pad_left
            t = torch.nn.functional.pad(t, (pad_left, pad_right, pad_top, pad_bot), value=0.5)

        # 5b. Rotation via vectorized horizontal shear (MPS-native)
        # Uses gather-based sub-pixel shifting — no Python row loops.
        # Single shear approximation: for ±20° the error vs true rotation
        # is <2% and the patch learns to handle the residual.
        angle_deg = random.uniform(-EOT_ROT_RANGE, EOT_ROT_RANGE)
        if abs(angle_deg) > 0.5:
            angle_rad = math.radians(angle_deg)
            _cur_h, _cur_w = t.shape[2], t.shape[3]
            shx = math.tan(angle_rad)  # horizontal displacement per row
            # Per-row fractional shift
            row_shifts = (torch.arange(_cur_h, device=device, dtype=torch.float32) - _cur_h / 2.0) * shx
            # Build source x-indices for each (row, col): src_col = col - shift[row]
            col_idx = torch.arange(_cur_w, device=device, dtype=torch.float32).unsqueeze(0)  # (1, W)
            src_x = col_idx - row_shifts.unsqueeze(1)  # (H, W)
            # Bilinear: blend floor and ceil
            src_x0 = src_x.long()
            frac = (src_x - src_x0.float()).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            # Clamp indices to valid range (out-of-bounds → edge pixel, masked by frac weighting)
            idx0 = src_x0.clamp(0, _cur_w - 1).unsqueeze(0).expand(1, 3, _cur_h, _cur_w)  # (1,3,H,W)
            idx1 = (src_x0 + 1).clamp(0, _cur_w - 1).unsqueeze(0).expand(1, 3, _cur_h, _cur_w)
            val0 = torch.gather(t, 3, idx0)
            val1 = torch.gather(t, 3, idx1)
            t = val0 * (1 - frac) + val1 * frac
            # Zero out pixels that came from out-of-bounds
            valid = (src_x >= 0) & (src_x < _cur_w - 1)  # (H, W)
            t = t * valid.unsqueeze(0).unsqueeze(0).float()

        # 5c. Perspective via vectorized resampling (MPS-native)
        # Two independent axes — vertical trapezoid (looking up/down) and
        # horizontal trapezoid (looking from the side, quilt wrapping around body).
        # Each uses gather-based sub-pixel column/row resampling.

        # Vertical perspective: per-row horizontal scaling
        persp_v = random.uniform(-EOT_PERSP_JITTER, EOT_PERSP_JITTER)
        if abs(persp_v) > 0.02:
            _cur_h, _cur_w = t.shape[2], t.shape[3]
            row_scales = torch.linspace(1.0 + persp_v, 1.0 - persp_v, _cur_h, device=device)
            centre_x = _cur_w / 2.0
            col_idx = torch.arange(_cur_w, device=device, dtype=torch.float32).unsqueeze(0)
            src_x = centre_x + (col_idx - centre_x) / row_scales.unsqueeze(1)
            src_x0 = src_x.long()
            frac = (src_x - src_x0.float()).unsqueeze(0).unsqueeze(0)
            idx0 = src_x0.clamp(0, _cur_w - 1).unsqueeze(0).expand(1, 3, _cur_h, _cur_w)
            idx1 = (src_x0 + 1).clamp(0, _cur_w - 1).unsqueeze(0).expand(1, 3, _cur_h, _cur_w)
            val0 = torch.gather(t, 3, idx0)
            val1 = torch.gather(t, 3, idx1)
            t = val0 * (1 - frac) + val1 * frac
            valid = (src_x >= 0) & (src_x < _cur_w - 1)
            t = t * valid.unsqueeze(0).unsqueeze(0).float() + \
                0.5 * (~valid).unsqueeze(0).unsqueeze(0).float()

        # Horizontal perspective: per-column vertical scaling
        # Simulates viewing the quilt from the side — one edge closer than the other
        persp_h = random.uniform(-EOT_PERSP_JITTER, EOT_PERSP_JITTER)
        if abs(persp_h) > 0.02:
            _cur_h, _cur_w = t.shape[2], t.shape[3]
            col_scales = torch.linspace(1.0 + persp_h, 1.0 - persp_h, _cur_w, device=device)
            centre_y = _cur_h / 2.0
            row_idx = torch.arange(_cur_h, device=device, dtype=torch.float32).unsqueeze(1)
            src_y = centre_y + (row_idx - centre_y) / col_scales.unsqueeze(0)
            src_y0 = src_y.long()
            frac = (src_y - src_y0.float()).unsqueeze(0).unsqueeze(0)
            idx0 = src_y0.clamp(0, _cur_h - 1).unsqueeze(0).expand(1, 3, _cur_h, _cur_w)
            idx1 = (src_y0 + 1).clamp(0, _cur_h - 1).unsqueeze(0).expand(1, 3, _cur_h, _cur_w)
            val0 = torch.gather(t, 2, idx0)
            val1 = torch.gather(t, 2, idx1)
            t = val0 * (1 - frac) + val1 * frac
            valid = (src_y >= 0) & (src_y < _cur_h - 1)
            t = t * valid.unsqueeze(0).unsqueeze(0).float() + \
                0.5 * (~valid).unsqueeze(0).unsqueeze(0).float()

        # 5d. Random translation (off-centre framing)
        max_tx = int(EOT_PERSP_JITTER * W * 0.3)
        max_ty = int(EOT_PERSP_JITTER * H * 0.3)
        if max_tx > 0 and max_ty > 0:
            tx = random.randint(-max_tx, max_tx)
            ty = random.randint(-max_ty, max_ty)
            t = torch.roll(t, shifts=(ty, tx), dims=(2, 3))

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

    # 9. JPEG simulation via real OpenCV encode/decode + straight-through estimator.
    #    Forward pass sees genuine JPEG block artifacts; backward flows through the
    #    clean tensor so gradients reach patch pixels unobstructed.
    quality = random.randint(EOT_JPEG_QUALITY[0], EOT_JPEG_QUALITY[1])
    with torch.no_grad():
        _arr = (t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        _arr_bgr = cv2.cvtColor(_arr, cv2.COLOR_RGB2BGR)
        _, _enc = cv2.imencode(".jpg", _arr_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        _arr_dec = cv2.cvtColor(cv2.imdecode(_enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        t_jpeg = torch.from_numpy(_arr_dec.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(t.device)
    t = t_jpeg.detach() + (t - t.detach())  # STE: forward=JPEG, backward=clean

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


def wrinkle_deform(patch_t: torch.Tensor, strength: float = EOT_WRINKLE_STRENGTH) -> torch.Tensor:
    """
    Apply a random smooth displacement to the patch tensor, simulating cloth
    wrinkles / fabric deformation.  MPS-native — uses per-row circular shifts
    with smooth interpolation instead of grid_sample.

    A smooth horizontal displacement curve (low-freq sinusoid + noise) shifts
    each row independently, simulating vertical wrinkle lines on fabric.
    Alternates between horizontal and vertical wrinkle orientation randomly.

    Parameters
    ----------
    patch_t : (1, 3, H, W) float32 tensor
    strength : max displacement as a fraction of H/W (default 0.08)

    Returns
    -------
    Warped patch tensor with same shape, gradients flow back to patch_t.
    """
    _, _, H, W = patch_t.shape
    if H < 4 or W < 4:
        return patch_t
    device = patch_t.device

    # Generate smooth per-row displacement via low-freq random signal
    # 4 control points → interpolate to H rows → smooth wrinkle-like shifts
    n_ctrl = 4
    ctrl = (torch.rand(n_ctrl, device=device) * 2 - 1) * strength
    # Interpolate to full resolution
    ctrl_up = torch.nn.functional.interpolate(
        ctrl.view(1, 1, 1, n_ctrl), size=(1, H), mode='bilinear', align_corners=False
    ).view(H)  # (H,) smooth displacement in fraction-of-width

    # Convert to pixel shifts
    shifts_px = ctrl_up * W  # (H,) float pixel shifts per row

    # Apply sub-pixel horizontal shift per row via weighted blend of two integer rolls
    shifts_int = shifts_px.long()
    shifts_frac = (shifts_px - shifts_int.float()).view(1, 1, H, 1)  # (1,1,H,1)

    # Roll each row — use two integer neighbours for sub-pixel blending
    # This is differentiable w.r.t. patch_t (linear blend), not w.r.t. shifts (constant)
    out = torch.zeros_like(patch_t)
    for r in range(H):
        s = int(shifts_int[r].item())
        f = shifts_frac[0, 0, r, 0]
        rolled_0 = torch.roll(patch_t[:, :, r, :], shifts=s, dims=-1)
        rolled_1 = torch.roll(patch_t[:, :, r, :], shifts=s + 1, dims=-1)
        out[:, :, r, :] = rolled_0 * (1 - f) + rolled_1 * f

    # 50% chance: apply vertically instead (transpose → shift → transpose back)
    if random.random() < 0.5:
        # Redo with column shifts instead
        ctrl2 = (torch.rand(n_ctrl, device=device) * 2 - 1) * strength
        ctrl_up2 = torch.nn.functional.interpolate(
            ctrl2.view(1, 1, 1, n_ctrl), size=(1, W), mode='bilinear', align_corners=False
        ).view(W)
        shifts_px2 = ctrl_up2 * H
        shifts_int2 = shifts_px2.long()
        shifts_frac2 = (shifts_px2 - shifts_int2.float()).view(1, 1, 1, W)
        out2 = torch.zeros_like(out)
        for c in range(W):
            s = int(shifts_int2[c].item())
            f = shifts_frac2[0, 0, 0, c]
            rolled_0 = torch.roll(out[:, :, :, c], shifts=s, dims=-1)
            rolled_1 = torch.roll(out[:, :, :, c], shifts=s + 1, dims=-1)
            out2[:, :, :, c] = rolled_0 * (1 - f) + rolled_1 * f
        return out2

    return out


def composite_leg_patch(
    host_t: torch.Tensor,        # (1, 3, H, W) float32 [0,1]
    patch: torch.Tensor,         # (1, 3, N, N) float32 [0,1]  requires_grad
    hip: np.ndarray,             # (2,) float32 pixel coords
    ankle: np.ndarray,           # (2,) float32 pixel coords
    img_h: int,
    img_w: int,
    width_frac: float = 0.25,
    geo_prob: float = 1.0,
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

    # --- optional cloth-wrinkle deformation (before resize) ------------------
    _p = patch
    if random.random() < EOT_WRINKLE_PROB * geo_prob:
        _p = wrinkle_deform(_p)

    # --- differentiable patch resize -----------------------------------------
    patch_resized = torch.nn.functional.interpolate(
        _p, size=(rect_h, rect_w), mode="bilinear", align_corners=False
    )  # (1, 3, rect_h, rect_w)

    # --- build mask and placement canvas (numpy, no grad) --------------------
    mask_np = make_leg_mask(hip, ankle, img_h, img_w, width_frac)
    mask_t = torch.from_numpy(mask_np).to(host_t.device).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    # Axis-aligned bounding box of the rotated rectangle — used to place the
    # resized patch onto the canvas before masking clips it to the true shape.
    center = (hip + ankle) / 2
    # Positional jitter: ±5% of leg length to prevent anchor-grid overfitting
    jitter_px = leg_length * EOT_POS_JITTER
    row = int(center[1] + random.uniform(-jitter_px, jitter_px)) - rect_h // 2
    col = int(center[0] + random.uniform(-jitter_px, jitter_px)) - rect_w // 2
    row = max(0, min(row, img_h - rect_h))
    col = max(0, min(col, img_w - rect_w))

    # Canvas: zero tensor with patch placed at AABB location — no in-place ops
    # so gradients can flow back through patch_resized to patch.
    ph = min(rect_h, img_h - row)
    pw = min(rect_w, img_w - col)
    patch_canvas = torch.zeros_like(host_t)
    # Use narrow + copy_ only on the patch_canvas (leaf-free), keeping graph intact
    # Build via addition of a padded patch tensor instead of in-place slice assignment
    pad_top    = row
    pad_bottom = img_h - row - ph
    pad_left   = col
    pad_right  = img_w - col - pw
    patch_placed = torch.nn.functional.pad(
        patch_resized[:, :, :ph, :pw],
        (pad_left, pad_right, pad_top, pad_bottom)
    )  # (1, 3, H, W) — fully differentiable

    composite = host_t * (1.0 - mask_t) + patch_placed * mask_t
    return composite, mask_t


def composite_torso_patch(
    host_t: torch.Tensor,       # (1, 3, H, W) float32 [0,1]
    patch_torso: torch.Tensor,  # (1, 3, N, N) float32 [0,1]  requires_grad
    kpts: np.ndarray,           # (17, 3) x, y, conf
    img_h: int,
    img_w: int,
    geo_prob: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Tile identical square patches in a 2D quilt over the torso (shoulder → hip),
    with random horizontal and vertical offset to simulate the quilt not being
    perfectly centred.  Wrapping ensures partial tiles are filled from the
    opposite edge of the pattern — like sliding a window over an infinite
    repeating quilt.

    Each tile is shoulder-width × shoulder-width.  The grid covers the full
    torso rectangle (shoulder-to-shoulder horizontally, shoulders-to-hips
    vertically).  A random x-offset in [0, side) shifts the grid each step
    so the patch learns to work at every possible seam alignment.

    Keypoint indices (COCO-17):
        5 = left shoulder,  6  = right shoulder
        11 = left hip,      12 = right hip

    Returns (composite, mask_t) where mask_t is (1,1,H,W).
    """
    ls_x, ls_y = float(kpts[5][0]),  float(kpts[5][1])
    rs_x, rs_y = float(kpts[6][0]),  float(kpts[6][1])
    lh_x, lh_y = float(kpts[11][0]), float(kpts[11][1])
    rh_x, rh_y = float(kpts[12][0]), float(kpts[12][1])

    # Tile dimensions: square, side = shoulder-to-shoulder width
    sw = max(ls_x, rs_x) - min(ls_x, rs_x)
    side = max(4, int(sw))

    # Expand torso box 20% beyond shoulders on each side so the quilt
    # covers the full upper-body width (sleeves / sides of garment).
    margin = sw * 0.20

    # Torso bounding box — no positional jitter needed here because the
    # quilt tile offset (torch.roll) already provides full anchor-grid coverage.
    torso_x1 = int(max(0,     min(ls_x, rs_x) - margin))
    torso_x2 = int(min(img_w, max(ls_x, rs_x) + margin))
    torso_y1 = int(max(0,     min(ls_y, rs_y)))
    torso_y2 = int(min(img_h, max(lh_y, rh_y)))

    torso_w = torso_x2 - torso_x1
    torso_h = torso_y2 - torso_y1
    # Skip if torso is too small — noisy keypoints on far-away or extreme
    # side-angle people produce bad gradients.  30px shoulder width is the
    # minimum for a tile that can carry meaningful adversarial features,
    # especially after EOT scaling (0.20x far → 6px, marginal but acceptable).
    if sw < 30 or torso_w < 30 or torso_h < 30:
        return host_t, torch.zeros(1, 1, img_h, img_w, device=host_t.device)

    # Resize the canonical patch to one tile at full side × side
    tile_full = torch.nn.functional.interpolate(
        patch_torso, size=(side, side), mode="bilinear", align_corners=False
    )  # (1, 3, side, side)

    # Random horizontal and vertical offset: simulates quilt not being centred.
    # The offset is in [0, side) — wrapping fills partial tiles from the
    # opposite edge of the pattern like an infinite repeating quilt.
    x_offset = random.randint(0, side - 1)
    y_offset = random.randint(0, side - 1)

    # Build a tiled texture that covers the full torso rectangle.
    # We create a texture buffer slightly larger than torso, tile into it,
    # then crop to exact torso size.
    n_tiles_x = (torso_w + side - 1) // side + 1  # +1 for partial offset tile
    n_tiles_y = (torso_h + side - 1) // side + 1
    buf_w = n_tiles_x * side
    buf_h = n_tiles_y * side

    # Tile by repeating the full-size tile across the buffer
    row_strip = tile_full.expand(1, 3, side, side).repeat(1, 1, 1, n_tiles_x)  # (1,3,side,buf_w)
    tiled = row_strip.repeat(1, 1, n_tiles_y, 1)  # (1, 3, buf_h, buf_w)

    # Apply offset (circular shift) then crop to torso dimensions
    tiled = torch.roll(tiled, shifts=(-y_offset, -x_offset), dims=(2, 3))
    tiled_cropped = tiled[:, :, :torso_h, :torso_w]  # (1, 3, torso_h, torso_w)

    # Optional cloth-wrinkle deformation applied to the assembled quilt so
    # wrinkles span across tile boundaries naturally, like real fabric.
    # Guarded by geo_prob to respect the geometric curriculum (grid_sample
    # backward is not implemented on MPS).
    if random.random() < EOT_WRINKLE_PROB * geo_prob:
        tiled_cropped = wrinkle_deform(tiled_cropped)

    # Pad cropped tile texture to full image size
    patch_placed = torch.nn.functional.pad(
        tiled_cropped,
        (torso_x1, img_w - torso_x2, torso_y1, img_h - torso_y2)
    )  # (1, 3, H, W)

    mask_np = np.zeros((img_h, img_w), dtype=np.float32)
    mask_np[torso_y1:torso_y2, torso_x1:torso_x2] = 1.0
    mask_t = torch.from_numpy(mask_np).to(host_t.device).unsqueeze(0).unsqueeze(0)

    composite = host_t * (1.0 - mask_t) + patch_placed * mask_t
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

        # --- Composite patch onto the CLEAN host (torso quilt) -----------
        kpts_list = pose_keypoints[idx] if idx < len(pose_keypoints) else []
        comp_ready = False
        if len(kpts_list) > 0:
            person_kpts = kpts_list[0]
            if person_kpts[5][2] >= 0.3 and person_kpts[6][2] >= 0.3 and person_kpts[11][2] >= 0.3:
                host_t = preprocess(orig_bgr)
                with torch.no_grad():
                    comp_t, _ = composite_torso_patch(
                        host_t, patch_t.cpu(), person_kpts, img_h, img_w
                    )
                comp_ready = True

        if comp_ready:
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

        # --- Post-patch saliency (torso quilt) ---------------------------
        kpts_list = pose_keypoints[idx] if idx < len(pose_keypoints) else []
        comp_ready = False
        if len(kpts_list) > 0:
            pk = kpts_list[0]
            if pk[5][2] >= 0.3 and pk[6][2] >= 0.3 and pk[11][2] >= 0.3:
                with torch.no_grad():
                    comp_t, mask_t = composite_torso_patch(
                        host_t, patch_t.cpu(), pk, img_h, img_w
                    )
                comp_ready = True

        if comp_ready:
            sal_post = _compute_saliency(comp_t, torch_model)
            comp_bgr = cv2.cvtColor(
                (comp_t.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                cv2.COLOR_RGB2BGR
            )
            # Draw torso outline on right panel
            mask_np = mask_t.squeeze().numpy()
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
    global _log_fh
    args = parse_args()
    if args.log_file:
        _log_fh = open(args.log_file, "w", buffering=1)  # line-buffered
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    # LEG PATCH PIPELINE ONLY
    # Remove all non-leg-patch logic and variables
    patch_size = args.patch_size
    # Always resolve out_path to an absolute path anchored at PROJECT_ROOT so that
    # the checkpoint path (derived from out_path) is stable regardless of CWD.
    if args.out:
        _op = Path(args.out)
        out_path = _op if _op.is_absolute() else PROJECT_ROOT / _op
    else:
        out_path = PROJECT_ROOT / "patterns" / f"patch_{patch_size}_{args.init}.png"
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
    best_loss     = float("inf")
    best_obj_loss = float("inf")   # adversarial objective only (no NPS/TV)
    best_mean_conf = float("inf")  # best cross-dataset eval confidence (lower = better)
    patch = init_patch(args.init, patch_size, device)
    patch.requires_grad_(True)
    best_patch = patch.detach().clone()

    # Dual-patch: torso patch trained jointly with leg patch
    patch_torso: torch.Tensor | None = None
    best_patch_torso: torch.Tensor | None = None
    if args.dual_patch:
        patch_torso = init_patch(args.init, patch_size, device)
        patch_torso.requires_grad_(True)
        best_patch_torso = patch_torso.detach().clone()
        print(f"[INFO] Dual-patch mode ON — torso patch ({patch_size}×{patch_size}) trained jointly")

    if args.resume:
        print(f"[INFO] Resume requested — looking for checkpoint: {ckpt_path}")
        if not ckpt_path.exists():
            print(f"[WARN] No checkpoint found at {ckpt_path}. Starting fresh.")
        else:
            try:
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                patch      = ckpt["patch"].to(device).requires_grad_(True)
                start_step = ckpt["step"]
                best_loss  = ckpt["best_loss"]
                best_patch = ckpt["best_patch"].to(device)
                best_mean_conf = ckpt.get("best_mean_conf", float("inf"))
                if args.dual_patch and "patch_torso" in ckpt:
                    patch_torso      = ckpt["patch_torso"].to(device).requires_grad_(True)
                    best_patch_torso = ckpt.get("best_patch_torso", patch_torso.detach().clone()).to(device)
                    print(f"[INFO] Resumed torso patch from checkpoint")
                print(f"[INFO] Resumed from checkpoint  step={start_step}  best_loss={best_loss:.6f}  best_mean_conf={best_mean_conf:.4f}")
            except Exception as e:
                print(f"[WARN] Failed to load checkpoint ({e}). Starting fresh.")

    # Hat/crown patch — separate tensor, same size as torso patch.
    # Stored and updated independently so gradients are clean.
    hat_patch: torch.Tensor | None = None
    best_hat_patch: torch.Tensor | None = None
    # Hat patch forcibly disabled; do not allocate

    # ------------------------------------------------------------------
    # 5. Baseline detections + pose keypoints (cached across runs)
    # ------------------------------------------------------------------
    _cache_key = _host_cache_key(args.hosts_dir, args.host, args.model)
    _cached = _load_startup_cache(_cache_key)

    if _cached is not None and len(_cached["pose_keypoints"]) == len(host_pool_bgr):
        print(f"[CACHE] Hit — reusing baseline detections + pose keypoints")
        baseline_confs = _cached["baseline_confs"]
        host_bboxes    = _cached["host_bboxes"]
        pose_keypoints = _cached["pose_keypoints"]
    else:
        if _cached is not None:
            print(f"[CACHE] Stale (host count changed) — recomputing")
        else:
            print(f"[CACHE] Miss — computing baseline + pose from scratch")

        # 5a. Baseline: average max-confidence person detection, all hosts, no patch
        print(f"[INFO] Computing clean baseline confidence ({len(host_pool_bgr)} hosts, no patch) …")
        baseline_confs = []
        host_bboxes: list = []  # cached person bbox per host for IoU-guided loss
        for _h_bgr in host_pool_bgr:
            with torch.no_grad():
                _res = yolo.predict(source=_h_bgr, conf=0.01, classes=[PERSON_CLASS], verbose=False)
            _boxes = _res[0].boxes
            if _boxes is not None and len(_boxes) > 0:
                baseline_confs.append(float(_boxes.conf.cpu().max()))
                _best_box_idx = int(_boxes.conf.cpu().argmax())
                _xyxy = _boxes.xyxy[_best_box_idx].cpu().numpy()
                host_bboxes.append((int(_xyxy[0]), int(_xyxy[1]), int(_xyxy[2]), int(_xyxy[3])))
            else:
                host_bboxes.append(None)

        # 5b. Pre-extract pose keypoints for all host images
        print("[INFO] Extracting pose keypoints from host images …")
        pose_model = YOLO("yolov8n-pose.pt")
        pose_keypoints: list = []
        for h_bgr in host_pool_bgr:
            results = pose_model(h_bgr, verbose=False)
            kpts = results[0].keypoints.data.cpu().numpy() if results[0].keypoints is not None else []
            pose_keypoints.append(kpts)
        print(f"[INFO] Pose extraction done  ({len(pose_keypoints)} images)")
        del pose_model  # free memory

        _save_startup_cache(_cache_key, baseline_confs, host_bboxes, pose_keypoints)

    clean_baseline = float(np.mean(baseline_confs)) if baseline_confs else 0.0
    print(f"[INFO] Clean baseline mean confidence : {clean_baseline:.6f}  ({len(baseline_confs)}/{len(host_pool_bgr)} hosts detected)")
    torch_model.eval().to(device)  # yolo.predict() may move model to CPU; restore
    yolo.overrides["device"] = str(device)

    # Pre-compute anchor grid once — constant for all steps
    anchor_centers_t = generate_anchor_centers(IMG_SIZE, device=device) if args.iou_loss else None
    if anchor_centers_t is not None:
        print(f"[INFO] IoU-guided loss enabled — anchor grid: {anchor_centers_t.shape}")

    # ------------------------------------------------------------------
    # 6. PGD loop with EOT augmentation + multi-host + leg-patch placement
    # ------------------------------------------------------------------
    for step in range(start_step + 1, args.steps + 1):
        if patch.grad is not None:
            patch.grad.zero_()
        if patch_torso is not None and patch_torso.grad is not None:
            patch_torso.grad.zero_()

        obj_losses = []
        attn_losses = []
        _torso_updates = 0  # count how many batch samples got a torso composite this step
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

            # (c) Differentiable composite
            host_t = host_pool_t[idx]

            # --- Torso mode: place patch on shoulder→hip region ---
            _conf_ls = person_kpts[5][2]
            _conf_rs = person_kpts[6][2]
            _conf_lh = person_kpts[11][2]
            if _conf_ls < 0.3 or _conf_rs < 0.3 or _conf_lh < 0.3:
                continue
            composite, mask_t = composite_torso_patch(
                host_t, patch, person_kpts, IMG_SIZE, IMG_SIZE, geo_prob=geo_prob
            )

            # (d) EOT augmentation
            # Far-view scale only when shoulder width >= 80px so the patch
            # stays above ~16px after 0.20× scaling (meaningful gradient).
            _sw = float(person_kpts[6][0]) - float(person_kpts[5][0])
            _sw = abs(_sw)
            # Tiered gate: medium band needs ≥50px shoulders (~8m),
            # far band needs ≥80px (~5m) so tiles stay above stride-8 after scaling.
            _allow_far = _sw >= 80.0
            _allow_med = _sw >= 50.0
            if not args.no_eot:
                composite_aug = eot_augment_differentiable(composite, geo_prob=geo_prob, allow_far_view=_allow_far, allow_med_view=_allow_med)
            else:
                composite_aug = composite

            if composite_aug.shape[-2:] != (IMG_SIZE, IMG_SIZE):
                composite_aug = torch.nn.functional.interpolate(
                    composite_aug, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
                )

            # (e) Forward loss
            _iou_bbox = host_bboxes[idx] if (args.iou_loss and idx < len(host_bboxes)) else None
            sample_loss = forward_person_loss(
                torch_model, composite_aug, args.topk, device,
                bbox=_iou_bbox, anchor_centers=anchor_centers_t, iou_sigma=args.iou_sigma
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
                sal_in  = sal_weight * mask_aug
                sal_out = sal_weight * (1.0 - mask_aug)
                L_in  = (sal_in  * composite_aug).sum() / (sal_in.sum()  + eps_mask)
                L_out = (sal_out * composite_aug).sum() / (sal_out.sum() + eps_mask)
                attn_loss = -(L_in - args.lambda_attn * L_out)
                attn_losses.append(attn_loss)

        if not obj_losses:
            continue  # no valid samples this step (all hosts lacked confident keypoints)

        # (f) Mean loss + worst-case term + regularisation
        #     The worst-case (max) term forces gradients toward the hardest
        #     image in the batch, preventing the patch from specialising on
        #     easy samples while ignoring resistant ones.
        loss = sum(obj_losses) / len(obj_losses) + 0.5 * max(obj_losses)
        if attn_losses:
            loss = loss + sum(attn_losses) / len(attn_losses)
        if args.alpha > 0 and printable_colors is not None:
            loss = loss + args.alpha * nps_loss(patch, printable_colors)
            if patch_torso is not None:
                loss = loss + args.alpha * nps_loss(patch_torso, printable_colors)
        if args.beta > 0:
            loss = loss + args.beta * tv_loss(patch)
            if patch_torso is not None:
                loss = loss + args.beta * tv_loss(patch_torso)

        # (g) Backward + PGD step (both patches updated jointly)
        loss.backward()
        with torch.no_grad():
            patch.data -= lr_curr * patch.grad.sign()
            patch.data.clamp_(0.0, 1.0)
        patch.grad = None
        if patch_torso is not None and patch_torso.grad is not None:
            with torch.no_grad():
                patch_torso.data -= lr_curr * patch_torso.grad.sign()
                patch_torso.data.clamp_(0.0, 1.0)
            patch_torso.grad = None

        if loss.item() < best_loss:
            best_loss = loss.item()
        # Track best batch obj_loss — used as fallback checkpoint selection
        # until the first CONF eval runs, then CONF eval takes over.
        _obj_loss_val = (sum(obj_losses) / len(obj_losses)).item()
        if _obj_loss_val < best_obj_loss:
            best_obj_loss = _obj_loss_val
            if best_mean_conf == float("inf"):
                # No CONF eval yet — use batch loss as fallback
                best_patch = patch.detach().clone()

        # Step-1 gradient sanity check: if the patch has no variation after the
        # first update, gradients are not flowing — abort early rather than waste time.
        if step == start_step + 1:
            with torch.no_grad():
                _std = patch.std().item()
                _mn  = patch.min().item()
                _mx  = patch.max().item()
            _log(f"[SANITY] Step 1 torso patch stats: std={_std:.4f}  range=[{_mn:.3f}, {_mx:.3f}]")
            if _std < 1e-4:
                _log("[ERROR] Torso patch has zero variation after step 1 — gradients are NOT flowing.")
                _log("[ERROR] Aborting. Check composite_torso_patch() for in-place ops.")
                raise RuntimeError("Gradient flow check failed: torso patch std < 1e-4 after step 1")
            else:
                _log("[SANITY] Torso patch gradient flow OK.")
            if patch_torso is not None:
                with torch.no_grad():
                    _ts = patch_torso.std().item()
                    _tmn = patch_torso.min().item()
                    _tmx = patch_torso.max().item()
                _log(f"[SANITY] Step 1 torso patch stats: std={_ts:.4f}  range=[{_tmn:.3f}, {_tmx:.3f}]  torso_updates={_torso_updates}")
                if _ts < 1e-4 and _torso_updates > 0:
                    _log("[ERROR] Torso patch not updating despite valid composites — check composite_torso_patch() for in-place ops.")
                elif _torso_updates == 0:
                    _log("[WARN]  Torso patch got 0 updates on step 1 — no images had both shoulders visible (kpts 5,6,11 conf>=0.3).")
                else:
                    _log("[SANITY] Torso patch gradient flow OK.")

        # Periodic checkpoint
        if step % args.checkpoint_every == 0:
            _ckpt_data = {
                "step":       step,
                "patch":      patch.detach().cpu(),
                "best_patch": best_patch.cpu(),
                "best_loss":  best_loss,
                "best_mean_conf": best_mean_conf,
            }
            if patch_torso is not None:
                _ckpt_data["patch_torso"]      = patch_torso.detach().cpu()
                _ckpt_data["best_patch_torso"] = best_patch_torso.cpu() if best_patch_torso is not None else patch_torso.detach().cpu()
            torch.save(_ckpt_data, ckpt_path)

        if args.verbose and step % 10 == 0:
            attn_str = f"  attn={sum(l.item() for l in attn_losses)/len(attn_losses):.4f}" if attn_losses else ""
            with torch.no_grad():
                _p_std = patch.std().item()
                _p_mn  = patch.min().item()
                _p_mx  = patch.max().item()
            patch_str = f"  patch_std={_p_std:.4f} [{_p_mn:.3f},{_p_mx:.3f}]"
            torso_str = f"  torso_upd={_torso_updates}/{args.batch_size}" if args.dual_patch else ""
            _log(f"  Step {step:>5d}/{args.steps}  loss={loss.item():.6f}  best_obj={best_obj_loss:.6f}  lr={lr_curr:.5f}  geo={geo_prob:.2f}{attn_str}{patch_str}{torso_str}")

        # Periodic mean-confidence check — same metric as baseline
        _conf_interval = getattr(args, "conf_interval", 200)
        if step > start_step and step % _conf_interval == 0:
            # Move model to CPU for yolo.predict() calls, restore after
            torch_model.eval().to("cpu")
            yolo.overrides["device"] = "cpu"
            _conf_check = []
            _bp_cpu = best_patch.detach().cpu()
            _bp_torso_cpu = best_patch_torso.detach().cpu() if (args.dual_patch and best_patch_torso is not None) else None
            for _ci, _h_bgr in enumerate(host_pool_bgr):
                _ckpts = pose_keypoints[_ci] if _ci < len(pose_keypoints) else []
                if len(_ckpts) > 0:
                    _pk = _ckpts[0]
                    _comp_c_ready = False
                    _host_ci = host_pool_t[_ci].cpu()
                    # Torso mode: composite on shoulder→hip region
                    _conf_ls_c = _pk[5][2]; _conf_rs_c = _pk[6][2]; _conf_lh_c = _pk[11][2]
                    if _conf_ls_c >= 0.3 and _conf_rs_c >= 0.3 and _conf_lh_c >= 0.3:
                        with torch.no_grad():
                            _comp_c, _ = composite_torso_patch(
                                _host_ci, _bp_cpu, _pk, IMG_SIZE, IMG_SIZE
                            )
                        _comp_c_ready = True
                    if _comp_c_ready:
                        _img_c = cv2.cvtColor(
                            (_comp_c.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                            cv2.COLOR_RGB2BGR
                        )
                        _res_c = yolo.predict(source=_img_c, conf=0.01, classes=[PERSON_CLASS], verbose=False)
                        _boxes_c = _res_c[0].boxes
                        _conf_check.append(float(_boxes_c.conf.cpu().max()) if (_boxes_c is not None and len(_boxes_c) > 0) else 0.0)
                    else:
                        # Can't apply patch (no valid keypoints) — use clean image
                        _res_c = yolo.predict(source=_h_bgr, conf=0.01, classes=[PERSON_CLASS], verbose=False)
                        _boxes_c = _res_c[0].boxes
                        if _boxes_c is not None and len(_boxes_c) > 0:
                            _conf_check.append(float(_boxes_c.conf.cpu().max()))
                else:
                    _res_c = yolo.predict(source=_h_bgr, conf=0.01, classes=[PERSON_CLASS], verbose=False)
                    _boxes_c = _res_c[0].boxes
                    if _boxes_c is not None and len(_boxes_c) > 0:
                        _conf_check.append(float(_boxes_c.conf.cpu().max()))
            torch_model.eval().to(device)  # restore to training device
            yolo.overrides["device"] = str(device)
            if _conf_check:
                _mean_c = float(np.mean(_conf_check))
                _suppressed_c = sum(1 for c in _conf_check if c < 0.25)
                _pct = 100.0 * (_mean_c - clean_baseline) / (clean_baseline + 1e-9)
                _log(f"[CONF]  Step {step:>5d}  mean_conf={_mean_c:.4f}  (baseline={clean_baseline:.4f}, Δ={_pct:+.1f}%)  suppressed(<0.25)={_suppressed_c}/{len(_conf_check)}")
                # Select best patch by cross-dataset eval rather than batch loss —
                # prevents specialisation on easy samples that skew best_obj.
                if _mean_c < best_mean_conf:
                    best_mean_conf = _mean_c
                    best_patch = patch.detach().clone()
                    if patch_torso is not None:
                        best_patch_torso = patch_torso.detach().clone()
                    _log(f"[CONF]  ↑ New best patch (mean_conf={_mean_c:.4f}, suppressed={_suppressed_c}/{len(_conf_check)})")

    # ------------------------------------------------------------------
    # 6b. Post-training summary
    # ------------------------------------------------------------------
    print(f"[INFO] Training complete. Best loss: {best_loss:.6f}")
    print("[INFO] Computing post-training confidence (best patch, all hosts) …")
    # Force Ultralytics to use CPU for predict() — manual .to("cpu") alone is
    # not enough because yolo.predict() internally re-routes to the original device.
    torch_model.eval().to("cpu")
    yolo.overrides["device"] = "cpu"
    final_confs = []
    _bp_cpu_final = best_patch.detach().cpu()
    _bp_torso_cpu_final = best_patch_torso.detach().cpu() if (args.dual_patch and best_patch_torso is not None) else None
    for _idx, _h_bgr in enumerate(host_pool_bgr):
        _kpts_list = pose_keypoints[_idx] if _idx < len(pose_keypoints) else []
        _comp_ready = False
        if len(_kpts_list) > 0:
            _pkpts = _kpts_list[0]
            _host_cpu = host_pool_t[_idx].cpu()
            with torch.no_grad():
                # Torso mode: composite on shoulder→hip region
                if _pkpts[5][2] >= 0.3 and _pkpts[6][2] >= 0.3 and _pkpts[11][2] >= 0.3:
                    _comp_t, _ = composite_torso_patch(
                        _host_cpu, _bp_cpu_final, _pkpts, IMG_SIZE, IMG_SIZE
                    )
                    _comp_ready = True
        if _comp_ready:
            _img_np = cv2.cvtColor(
                (_comp_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8),
                cv2.COLOR_RGB2BGR
            )
        else:
            _img_np = _h_bgr
        try:
            with torch.no_grad():
                _res = yolo.predict(source=_img_np, conf=0.01, classes=[PERSON_CLASS], verbose=False)
            _boxes = _res[0].boxes
            if _boxes is not None and len(_boxes) > 0:
                final_confs.append(float(_boxes.conf.cpu().max()))
            else:
                final_confs.append(0.0)
        except Exception as _eval_err:
            print(f"[WARN] Post-eval predict failed for host {_idx}: {_eval_err}")
            final_confs.append(0.0)
    clean_final = float(np.mean(final_confs)) if final_confs else 0.0
    reduction = (1.0 - clean_final / clean_baseline) * 100.0 if clean_baseline > 0 else 0.0
    print(f"[INFO] Confidence reduction : {clean_baseline:.6f} → {clean_final:.6f}  ({reduction:.1f}% suppression)")
    torch_model.eval().to(device)  # restore after predict() calls
    yolo.overrides["device"] = str(device)

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

    # Save torso patch (dual-patch mode)
    if args.dual_patch and best_patch_torso is not None:
        _torso_out = Path(args.torso_out) if args.torso_out else out_path.parent / (out_path.stem + "_torso.png")
        _torso_np  = best_patch_torso.squeeze(0).permute(1, 2, 0).cpu().numpy()
        _torso_bgr = cv2.cvtColor((_torso_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(_torso_out), _torso_bgr)
        cv2.imwrite(str(_torso_out.parent / (_torso_out.stem + "_preview.png")),
                    cv2.resize(_torso_bgr, (512, 512), interpolation=cv2.INTER_NEAREST))
        cv2.imwrite(str(iter_dir / _torso_out.name), _torso_bgr)
        print(f"[INFO] Torso patch → {_torso_out}")

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
