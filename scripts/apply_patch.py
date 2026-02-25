"""
apply_patch.py
==============
Expectation-over-Transformation (EOT) pipeline.

Loads the synthesised adversarial patch (patterns/patch_128.png) and composites
it onto every image in data/clean/, emulating the variation seen in physical-world
deployments:

  EOT transform stack (applied to the PATCH before compositing):
    1. Random isotropic rescale  … ±30 % of patch nominal size
    2. Random rotation           … ±15 °
    3. Random perspective warp   … subtle skew to simulate non-frontal placement
    4. Gaussian blur             … kernel 0–3 px  (print/ focus softening)
    5. Brightness & contrast jitter

Composited images are saved to data/adversarial/ preserving the original
filename so evaluate.py can pair them automatically.

Usage
-----
  python scripts/apply_patch.py
  python scripts/apply_patch.py --patch patterns/my_patch.png --n-aug 5
  python scripts/apply_patch.py --placement center
"""

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATCH   = str(PROJECT_ROOT / "patterns" / "patch_256.png")
DEFAULT_CLEAN   = str(PROJECT_ROOT / "data" / "clean")
DEFAULT_ADV     = str(PROJECT_ROOT / "data" / "adversarial")
SUPPORTED_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EOT patch applicator for adversarial robustness experiments")
    p.add_argument("--patch",     default=DEFAULT_PATCH, help="Adversarial patch PNG path")
    p.add_argument("--clean-dir", default=DEFAULT_CLEAN, help="Directory of clean test images")
    p.add_argument("--adv-dir",   default=None,
                   help="Output directory for adversarial images "
                        "(defaults to data/adversarial/<noise_type>/ auto-detected from patch filename)")
    p.add_argument("--placement", choices=["center", "random", "torso", "top-left", "top-right"],
                   default="torso", help="Where on the target image to place the patch (default: torso)")
    p.add_argument("--n-aug",  type=int, default=1,
                   help="Number of augmented copies produced per clean image (default 1)")
    p.add_argument("--display-size", type=int, default=160,
                   help="Resize patch to this pixel width before compositing (default 160). "
                        "Controls physical coverage on the target; EOT scale variation is applied on top.")
    p.add_argument("--scale-range", type=float, nargs=2, default=[0.7, 1.3],
                   metavar=("MIN", "MAX"), help="Relative scale range for EOT resize (default 0.7 1.3)")
    p.add_argument("--rot-range",   type=float, default=15.0,
                   help="Max rotation angle in degrees for EOT (default ±15°)")
    p.add_argument("--seed",  type=int, default=0, help="Random seed")
    return p.parse_args()


# ---------------------------------------------------------------------------
# EOT transform helpers
# ---------------------------------------------------------------------------

def random_scale(patch: np.ndarray, scale_range: tuple[float, float]) -> np.ndarray:
    """Randomly rescale the patch."""
    factor = random.uniform(*scale_range)
    h, w = patch.shape[:2]
    new_h = max(4, int(h * factor))
    new_w = max(4, int(w * factor))
    return cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def random_rotation(patch: np.ndarray, max_angle: float) -> np.ndarray:
    """Rotate patch by a random angle and keep the full bounding box."""
    angle = random.uniform(-max_angle, max_angle)
    h, w  = patch.shape[:2]
    cx, cy = w / 2, h / 2
    M  = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    # Compute new canvas size to avoid clipping corners
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    rotated = cv2.warpAffine(patch, M, (new_w, new_h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated


def random_perspective(patch: np.ndarray, strength: float = 0.05) -> np.ndarray:
    """Apply a subtle random perspective warp."""
    h, w = patch.shape[:2]
    def jitter():
        return random.uniform(-strength, strength)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [jitter() * w, jitter() * h],
        [w + jitter() * w, jitter() * h],
        [w + jitter() * w, h + jitter() * h],
        [jitter() * w, h + jitter() * h],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(patch, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def random_blur(patch: np.ndarray, max_kernel: int = 3) -> np.ndarray:
    """Apply Gaussian blur with a randomly chosen (odd) kernel size."""
    k = random.choice([1, 1, 1, 3, 3, max_kernel if max_kernel % 2 == 1 else max_kernel + 1])
    if k <= 1:
        return patch
    return cv2.GaussianBlur(patch, (k, k), 0)


def random_brightness_contrast(patch: np.ndarray) -> np.ndarray:
    """Jitter brightness (α) and contrast (β)."""
    alpha = random.uniform(0.7, 1.3)   # contrast
    beta  = random.uniform(-20, 20)    # brightness offset
    adjusted = cv2.convertScaleAbs(patch, alpha=alpha, beta=beta)
    return adjusted


def apply_eot(patch_bgr: np.ndarray, scale_range: tuple[float, float], rot_range: float) -> np.ndarray:
    """Apply the full EOT transform stack to the patch."""
    p = random_scale(patch_bgr, scale_range)
    p = random_rotation(p, rot_range)
    p = random_perspective(p, strength=0.04)
    p = random_blur(p, max_kernel=3)
    p = random_brightness_contrast(p)
    return p


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------

def compute_placement(
    img_h: int,
    img_w: int,
    patch_h: int,
    patch_w: int,
    mode: str,
) -> tuple[int, int]:
    """Return top-left (row, col) for patch placement."""
    if mode == "center":
        row = max(0, (img_h - patch_h) // 2)
        col = max(0, (img_w - patch_w) // 2)
    elif mode == "torso":
        # Upper-torso band: 30-55% from top, horizontally centred.
        # Mirrors the placement used during PGD patch generation.
        row = max(0, int(img_h * 0.30))
        col = max(0, (img_w - patch_w) // 2)
    elif mode == "random":
        row = random.randint(0, max(0, img_h - patch_h))
        col = random.randint(0, max(0, img_w - patch_w))
    elif mode == "top-left":
        row, col = 10, 10
    elif mode == "top-right":
        row = 10
        col = max(0, img_w - patch_w - 10)
    else:
        row = col = 0
    return row, col


def composite(image: np.ndarray, patch: np.ndarray, row: int, col: int) -> np.ndarray:
    """Paste the patch onto a copy of the image at (row, col)."""
    out = image.copy()
    ph = min(patch.shape[0], out.shape[0] - row)
    pw = min(patch.shape[1], out.shape[1] - col)
    if ph <= 0 or pw <= 0:
        return out
    out[row:row + ph, col:col + pw] = patch[:ph, :pw]
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    patch_path = Path(args.patch)
    if not patch_path.exists():
        raise FileNotFoundError(
            f"Patch not found: {patch_path}\n"
            "  → Run 'python scripts/generate_patch.py' first."
        )

    # Auto-detect noise type from patch filename
    # e.g. patch_384_blocky.png  →  "blocky"
    #      patch_256.png         →  "unknown"
    stem_parts = patch_path.stem.split("_")   # ["patch", "384", "blocky"]
    known_types = {"uniform", "gaussian", "checkerboard", "stripes",
                   "salt_pepper", "gray", "blocky", "perlin"}
    detected_noise = next(
        (p for p in reversed(stem_parts) if p in known_types), "unknown"
    )

    # Resolve adv-dir: explicit flag > auto-detected subdir
    if args.adv_dir:
        adv_dir = Path(args.adv_dir)
    else:
        adv_dir = PROJECT_ROOT / "data" / "adversarial" / detected_noise

    adv_dir.mkdir(parents=True, exist_ok=True)

    clean_dir = Path(args.clean_dir)
    images = [p for p in sorted(clean_dir.iterdir()) if p.suffix.lower() in SUPPORTED_EXTS and not p.name.startswith("._")]
    if not images:
        print(f"[WARN] No images found in {clean_dir}  (supported: {SUPPORTED_EXTS})")
        print("       Place test images in data/clean/ and re-run.")
        return

    patch_bgr = cv2.imread(str(patch_path))
    if patch_bgr is None:
        raise RuntimeError(f"Could not read patch: {patch_path}")

    INFERENCE_SIZE = 640   # must match YOLOv8 imgsz used during patch generation

    trained_size = patch_bgr.shape[1]

    # Resize patch to display size — separates training resolution from physical coverage
    if args.display_size and args.display_size != patch_bgr.shape[1]:
        scale = args.display_size / patch_bgr.shape[1]
        new_h = max(4, int(patch_bgr.shape[0] * scale))
        patch_bgr = cv2.resize(patch_bgr, (args.display_size, new_h), interpolation=cv2.INTER_LINEAR)

    print(f"[INFO] Patch      : {patch_path}  (trained {trained_size}px → display {args.display_size}px)")
    print(f"[INFO] Noise type : {detected_noise}")
    print(f"[INFO] Output dir : {adv_dir}")
    print(f"[INFO] Clean imgs : {len(images)}  in {clean_dir}")
    print(f"[INFO] Augmentations per image : {args.n_aug}")
    print(f"[INFO] Placement  : {args.placement}")
    print(f"[INFO] Display size: {args.display_size}px  ({args.display_size/INFERENCE_SIZE*100:.0f}% of {INFERENCE_SIZE}px image width)")
    print(f"[INFO] Pre-resize   : all images → {INFERENCE_SIZE}×{INFERENCE_SIZE} before compositing")

    generated = 0
    for img_path in images:
        img_raw = cv2.imread(str(img_path))
        if img_raw is None:
            print(f"[WARN] Skipping unreadable: {img_path.name}")
            continue
        # Resize to match the inference resolution used during patch optimisation.
        # This ensures the 128×128 patch covers the same proportion of the image
        # at inference time as it did when gradients were computed.
        img = cv2.resize(img_raw, (INFERENCE_SIZE, INFERENCE_SIZE), interpolation=cv2.INTER_LINEAR)

        for aug_idx in range(args.n_aug):
            # Apply EOT transforms to the patch
            transformed_patch = apply_eot(patch_bgr, tuple(args.scale_range), args.rot_range)

            # Determine placement
            row, col = compute_placement(
                img.shape[0], img.shape[1],
                transformed_patch.shape[0], transformed_patch.shape[1],
                args.placement,
            )

            # Composite and save
            adv_img = composite(img, transformed_patch, row, col)

            if args.n_aug == 1:
                out_name = img_path.name
            else:
                out_name = f"{img_path.stem}_aug{aug_idx:02d}{img_path.suffix}"

            out_path = adv_dir / out_name
            cv2.imwrite(str(out_path), adv_img)
            generated += 1

    print(f"[INFO] Saved {generated} adversarial image(s) → {adv_dir}")


if __name__ == "__main__":
    main()
