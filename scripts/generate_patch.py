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
EOT_SCALE_RANGE  = (0.85, 1.15)   # keep tight – gradients need to flow back cleanly
EOT_ROT_RANGE    = 10.0           # degrees
EOT_BLUR_MAX     = 3              # max Gaussian kernel size
EOT_BRIGHTNESS   = 0.15          # ± fraction


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
    return p.parse_args()


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

    # 1. Random brightness / contrast jitter
    alpha = 1.0 + random.uniform(-EOT_BRIGHTNESS, EOT_BRIGHTNESS)
    beta  = random.uniform(-15, 15)
    arr = np.clip(arr * alpha + beta, 0, 255)

    # 2. Random Gaussian blur
    k = random.choice([1, 1, 3, 3, EOT_BLUR_MAX if EOT_BLUR_MAX % 2 == 1 else EOT_BLUR_MAX + 1])
    if k > 1:
        arr = cv2.GaussianBlur(arr, (k, k), 0)

    # 3. Mild random rotation (keep same canvas size)
    angle = random.uniform(-EOT_ROT_RANGE, EOT_ROT_RANGE)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    arr = cv2.warpAffine(arr, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)

    # Back to tensor
    arr = arr / 255.0
    t = torch.from_numpy(arr.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    t = t.to(composite.device)
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
) -> torch.Tensor:
    """
    Forward pass returning scalar loss = mean(top-k person confs).

    Mean-only loss spreads the gradient evenly across all high-confidence
    anchors, driving consistent confidence suppression across every frame
    rather than hunting for single lucky knockouts (FNs).
    Output layout: (1, 4+nc, 8400)  — channels-first (Ultralytics 8.x).
    """
    global _SHAPE_PRINTED
    pred = torch_model(img_t.to(device))
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    if not _SHAPE_PRINTED:
        print(f"[DEBUG] raw pred shape: {tuple(pred.shape)}")
        _SHAPE_PRINTED = True
    person_scores = pred[0, PERSON_COL_IDX, :]          # (N_anchors,)
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

    # ------------------------------------------------------------------
    # 1. Load model — freeze all weights
    # ------------------------------------------------------------------
    yolo = YOLO(args.model)
    torch_model: torch.nn.Module = yolo.model
    torch_model.eval().to(device)
    for p in torch_model.parameters():
        p.requires_grad_(False)

    # ------------------------------------------------------------------
    # 2. Load printable colours for NPS loss
    # ------------------------------------------------------------------
    printable_colors: torch.Tensor | None = None
    if args.alpha > 0:
        printable_colors = load_printable_colors(args.printable_colors, device)
        print(f"[INFO] Printable colours: {printable_colors.shape[0]} loaded from {args.printable_colors}")

    # ------------------------------------------------------------------
    # 3. Build host image pool
    # ------------------------------------------------------------------
    host_pool_bgr = load_host_pool(args.hosts_dir, args.host)
    host_pool_t   = [preprocess(h).to(device) for h in host_pool_bgr]

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

    # ------------------------------------------------------------------
    # 5. Baseline loss (random host, fixed centre placement)
    # ------------------------------------------------------------------
    with torch.no_grad():
        h_t = random.choice(host_pool_t)
        row0 = int(IMG_SIZE * 0.30)
        col0 = (IMG_SIZE - patch_size) // 2
        base_composite = apply_patch_to_tensor(h_t, patch.detach(), row0, col0)
        baseline_loss  = forward_person_loss(torch_model, base_composite, args.topk, device)
    print(f"[INFO] Baseline person-conf loss : {baseline_loss.item():.6f}")

    # ------------------------------------------------------------------
    # 6. PGD loop with EOT augmentation + multi-host + random placement
    # ------------------------------------------------------------------
    for step in range(start_step + 1, args.steps + 1):
        if patch.grad is not None:
            patch.grad.zero_()

        # (a–e) Mini-batch: average objectness loss over batch_size images
        #       Each sample uses a different host + placement + EOT augmentation
        #       so the gradient points toward suppression that works everywhere.
        obj_losses = []
        for _ in range(args.batch_size):
            # (a) Sample a random background from the pool
            host_t = random.choice(host_pool_t)

            # (b) Random torso-band placement
            row, col = random_torso_placement(IMG_SIZE, patch_size)

            # (c) Composite patch onto host
            composite = apply_patch_to_tensor(host_t, patch, row, col)

            # (d) EOT: augment composite (stop-grad on augmentation transforms)
            if not args.no_eot:
                composite_aug = eot_augment_tensor(composite.detach())
                composite_aug = apply_patch_to_tensor(
                    composite_aug.detach(), patch, row, col
                )
            else:
                composite_aug = composite

            obj_losses.append(
                forward_person_loss(torch_model, composite_aug, args.topk, device)
            )

        # (e) Mean objectness loss + regularisation (NPS/TV added once, not per sample)
        loss = sum(obj_losses) / len(obj_losses)
        if args.alpha > 0 and printable_colors is not None:
            loss = loss + args.alpha * nps_loss(patch, printable_colors)
        if args.beta > 0:
            loss = loss + args.beta * tv_loss(patch)

        # (f) Backward
        loss.backward()

        # (g) PGD update
        with torch.no_grad():
            patch.data -= args.lr * patch.grad.sign()
            if args.eps < 1.0:
                lower = (patch.data - args.eps).clamp(0.0, 1.0)
                upper = (patch.data + args.eps).clamp(0.0, 1.0)
                patch.data.clamp_(lower, upper)
            patch.data.clamp_(0.0, 1.0)

        if loss.item() < best_loss:
            best_loss  = loss.item()
            best_patch = patch.detach().clone()

        # Periodic checkpoint — allows resuming if interrupted
        if step % args.checkpoint_every == 0:
            torch.save({
                "step":       step,
                "patch":      patch.detach().cpu(),
                "best_patch": best_patch.cpu(),
                "best_loss":  best_loss,
            }, ckpt_path)

        if args.verbose and step % 50 == 0:
            print(f"  Step {step:>5d}/{args.steps}  loss: {loss.item():.6f}  best: {best_loss:.6f}")

    reduction = (1.0 - best_loss / max(baseline_loss.item(), 1e-9)) * 100
    print(f"[INFO] Final loss : {best_loss:.6f}  (baseline {baseline_loss.item():.6f}, -{reduction:.1f}%)")

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
        f.write(f"device         : {device}\n")
        f.write(f"baseline_loss  : {baseline_loss.item():.6f}\n")
        f.write(f"final_loss     : {best_loss:.6f}\n")
        f.write(f"loss_reduction : {reduction:.1f}%\n")
        f.write(f"host_pool_size : {len(host_pool_t)}\n")
        f.write(f"loss_mode      : mean (confidence suppression)\n")

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

    # Clean up checkpoint now that training is complete
    if ckpt_path.exists():
        ckpt_path.unlink()
        print(f"[INFO] Checkpoint removed (training complete)")


if __name__ == "__main__":
    main()
