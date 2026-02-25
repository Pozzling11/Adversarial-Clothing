"""
compare_inits.py
================
Train one adversarial patch per noise-initialisation type, apply each to the
clean image set, evaluate, and print a ranked comparison table.

Inits tested: uniform, gaussian, checkerboard, stripes, salt_pepper, gray

Usage
-----
  python scripts/compare_inits.py
  python scripts/compare_inits.py --steps 1500 --patch-size 384 --n-aug 5
"""

import argparse
import csv
import random
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Import helpers from sibling scripts
from generate_patch import (          # noqa: E402
    IMG_SIZE, PERSON_COL_IDX, TOP_K,
    apply_patch_to_tensor,
    eot_augment_tensor,
    forward_person_loss,
    init_patch,
    load_host_pool,
    preprocess,
    random_torso_placement,
)
from apply_patch import (             # noqa: E402
    apply_eot,
    composite,
    compute_placement,
    SUPPORTED_EXTS,
)
from evaluate import detect_persons   # noqa: E402

from ultralytics import YOLO

INIT_MODES = ["uniform", "gaussian", "checkerboard", "stripes", "salt_pepper", "gray", "blocky", "perlin"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare adversarial patch init strategies")
    p.add_argument("--model",      default=str(PROJECT_ROOT / "yolov8n.pt"))
    p.add_argument("--hosts-dir",  default=str(PROJECT_ROOT / "data" / "clean"))
    p.add_argument("--clean-dir",  default=str(PROJECT_ROOT / "data" / "clean"))
    p.add_argument("--patch-size", type=int,   default=384)
    p.add_argument("--steps",      type=int,   default=1500)
    p.add_argument("--lr",         type=float, default=0.03)
    p.add_argument("--n-aug",      type=int,   default=5)
    p.add_argument("--conf",       type=float, default=0.25)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--no-eot",     action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Train one patch
# ---------------------------------------------------------------------------

def train_patch(
    torch_model: torch.nn.Module,
    host_pool_t: list,
    init_mode: str,
    patch_size: int,
    steps: int,
    lr: float,
    use_eot: bool,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, float, float]:
    """Returns (best_patch, baseline_loss, best_loss)."""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    patch = init_patch(init_mode, patch_size, device)
    patch.requires_grad_(True)

    with torch.no_grad():
        h_t  = random.choice(host_pool_t)
        row0 = int(IMG_SIZE * 0.30)
        col0 = (IMG_SIZE - patch_size) // 2
        base_comp   = apply_patch_to_tensor(h_t, patch.detach(), row0, col0)
        baseline    = forward_person_loss(torch_model, base_comp, TOP_K, device).item()

    best_loss  = float("inf")
    best_patch = patch.detach().clone()

    for _ in range(steps):
        if patch.grad is not None:
            patch.grad.zero_()

        host_t      = random.choice(host_pool_t)
        row, col    = random_torso_placement(IMG_SIZE, patch_size)
        comp        = apply_patch_to_tensor(host_t, patch, row, col)

        if use_eot:
            comp_aug = eot_augment_tensor(comp.detach())
            comp_aug = apply_patch_to_tensor(comp_aug.detach(), patch, row, col)
        else:
            comp_aug = comp

        loss = forward_person_loss(torch_model, comp_aug, TOP_K, device)
        loss.backward()

        with torch.no_grad():
            patch.data -= lr * patch.grad.sign()
            patch.data.clamp_(0.0, 1.0)

        if loss.item() < best_loss:
            best_loss  = loss.item()
            best_patch = patch.detach().clone()

    return best_patch, baseline, best_loss


# ---------------------------------------------------------------------------
# Apply + evaluate one patch
# ---------------------------------------------------------------------------

def apply_and_evaluate(
    model: YOLO,
    patch_bgr: np.ndarray,
    clean_images: list[Path],
    patch_size: int,
    n_aug: int,
    conf: float,
) -> dict:
    """Return summary stats dict for this patch."""
    INFERENCE_SIZE = 640
    clean_confs, adv_confs, fn_count, total_adv = [], [], 0, 0

    for img_path in clean_images:
        img_raw = cv2.imread(str(img_path))
        if img_raw is None:
            continue
        img_clean = cv2.resize(img_raw, (INFERENCE_SIZE, INFERENCE_SIZE))

        # Clean detection
        det, max_cf, _ = detect_persons(model, img_clean, conf, 0.45, INFERENCE_SIZE)
        if det:
            clean_confs.append(max_cf)

        # Adversarial augmentations
        for _ in range(n_aug):
            p_aug  = apply_eot(patch_bgr, (0.7, 1.3), 15.0)
            row, col = compute_placement(INFERENCE_SIZE, INFERENCE_SIZE,
                                         p_aug.shape[0], p_aug.shape[1], "torso")
            adv_img = composite(img_clean, p_aug, row, col)
            adv_det, adv_cf, _ = detect_persons(model, adv_img, conf, 0.45, INFERENCE_SIZE)
            adv_confs.append(adv_cf)
            total_adv += 1
            if det and not adv_det:
                fn_count += 1

    return {
        "clean_avg_conf": round(np.mean(clean_confs) if clean_confs else 0.0, 4),
        "adv_avg_conf":   round(np.mean(adv_confs)   if adv_confs   else 0.0, 4),
        "fn_count":       fn_count,
        "total_adv":      total_adv,
        "fn_rate":        round(fn_count / total_adv * 100 if total_adv else 0.0, 1),
        "conf_drop":      round(
            (np.mean(clean_confs) - np.mean(adv_confs)) / max(np.mean(clean_confs), 1e-9) * 100
            if clean_confs and adv_confs else 0.0, 1
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Device: {device}  |  Patch: {args.patch_size}px  |  "
          f"Steps: {args.steps}  |  LR: {args.lr}  |  EOT: {'OFF' if args.no_eot else 'ON'}")
    print(f"[INFO] Comparing {len(INIT_MODES)} init modes: {', '.join(INIT_MODES)}\n")

    # Load model once
    yolo = YOLO(args.model)
    torch_model = yolo.model
    torch_model.eval().to(device)
    for p in torch_model.parameters():
        p.requires_grad_(False)

    # Load host pool once
    host_pool_bgr = load_host_pool(args.hosts_dir, None)
    host_pool_t   = [preprocess(h).to(device) for h in host_pool_bgr]

    # Clean image list
    clean_dir = Path(args.clean_dir)
    clean_images = sorted(
        p for p in clean_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTS and not p.name.startswith("._")
    )

    # Patterns output dir
    patterns_dir = PROJECT_ROOT / "patterns"
    patterns_dir.mkdir(exist_ok=True)

    results = []

    for mode in INIT_MODES:
        print(f"{'='*60}")
        print(f"  Training — init: {mode}")
        print(f"{'='*60}")

        best_patch_t, baseline, final_loss = train_patch(
            torch_model, host_pool_t, mode,
            args.patch_size, args.steps, args.lr,
            not args.no_eot, device, args.seed,
        )

        reduction = (1.0 - final_loss / max(baseline, 1e-9)) * 100
        print(f"  Loss: {baseline:.6f} → {final_loss:.6f}  ({reduction:.1f}% reduction)")

        # Save patch PNG
        patch_np  = best_patch_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
        patch_bgr = cv2.cvtColor((patch_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        out_path  = patterns_dir / f"patch_{args.patch_size}_{mode}.png"
        cv2.imwrite(str(out_path), patch_bgr)

        # Save preview
        preview = patterns_dir / f"patch_{args.patch_size}_{mode}_preview.png"
        cv2.imwrite(str(preview), cv2.resize(patch_bgr, (512, 512), interpolation=cv2.INTER_NEAREST))

        # Apply + evaluate
        print(f"  Evaluating ({args.n_aug} EOT augs × {len(clean_images)} images) …")
        stats = apply_and_evaluate(
            yolo, patch_bgr, clean_images,
            args.patch_size, args.n_aug, args.conf,
        )

        row = {
            "init":           mode,
            "train_baseline": round(baseline, 6),
            "train_final":    round(final_loss, 6),
            "loss_reduction": f"{reduction:.1f}%",
            **stats,
        }
        results.append(row)

        print(f"  FN rate: {stats['fn_rate']}%  ({stats['fn_count']}/{stats['total_adv']})  "
              f"|  conf drop: {stats['conf_drop']}%  "
              f"|  adv avg conf: {stats['adv_avg_conf']}\n")

    # ------------------------------------------------------------------
    # Ranked table
    # ------------------------------------------------------------------
    results.sort(key=lambda r: (-r["fn_count"], r["adv_avg_conf"]))

    print(f"\n{'='*78}")
    print(f"  RANKED COMPARISON  (conf threshold: {args.conf})")
    print(f"{'='*78}")
    hdr = f"  {'Init':<14} {'FNs':>5} {'FN%':>6} {'ConfDrop%':>10} {'AdvConf':>9} {'LossRed':>9}"
    print(hdr)
    print(f"  {'-'*72}")
    for r in results:
        fn_tag = "  ← BEST" if r == results[0] else ""
        print(f"  {r['init']:<14} {r['fn_count']:>5}/{r['total_adv']:<4} "
              f"{r['fn_rate']:>5.1f}% {r['conf_drop']:>9.1f}% "
              f"{r['adv_avg_conf']:>9.4f} {r['loss_reduction']:>9}{fn_tag}")
    print(f"{'='*78}\n")

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = results_dir / f"init_comparison_{ts}.csv"
    fields = ["init", "train_baseline", "train_final", "loss_reduction",
              "clean_avg_conf", "adv_avg_conf", "fn_count", "total_adv",
              "fn_rate", "conf_drop"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"[INFO] Comparison CSV saved → {csv_path}")


if __name__ == "__main__":
    main()
