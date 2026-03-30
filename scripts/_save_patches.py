"""One-shot script to extract best and final patches from checkpoint."""
import torch, cv2, numpy as np
from pathlib import Path

ckpt = torch.load("patterns/patch_torso_repro_ckpt.pt", map_location="cpu", weights_only=False)
print(f"Checkpoint step: {ckpt['step']}, best_loss: {ckpt['best_loss']:.6f}")

def save_patch(tensor, name_base, label):
    t = tensor.squeeze(0)  # [3,H,W]
    arr = (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    p1 = f"patterns/{name_base}.png"
    cv2.imwrite(p1, bgr)
    big = cv2.resize(bgr, (bgr.shape[1] * 4, bgr.shape[0] * 4), interpolation=cv2.INTER_NEAREST)
    p2 = f"patterns/{name_base}_4x.png"
    cv2.imwrite(p2, big)
    print(f"  [{label}] saved {p1}  +  {p2}")
    return bgr

best_bgr  = save_patch(ckpt["best_patch"], "patch_torso_repro_best",  "BEST (step ~10540)")
final_bgr = save_patch(ckpt["patch"],      "patch_torso_repro_final", "FINAL (step 12000)")

# Save to iterations folder
iters = Path("patterns/iterations")
iters.mkdir(parents=True, exist_ok=True)
existing = sorted([d for d in iters.iterdir() if d.is_dir() and d.name.startswith("iter")])
next_num = len(existing) + 1
out_dir = iters / f"iter{next_num:02d}"
out_dir.mkdir(exist_ok=True)

for fname, bgr in [
    ("patch_best.png",    best_bgr),
    ("patch_best_4x.png", cv2.resize(best_bgr,  (best_bgr.shape[1]*4,  best_bgr.shape[0]*4),  interpolation=cv2.INTER_NEAREST)),
    ("patch_final.png",   final_bgr),
    ("patch_final_4x.png",cv2.resize(final_bgr, (final_bgr.shape[1]*4, final_bgr.shape[0]*4), interpolation=cv2.INTER_NEAREST)),
]:
    cv2.imwrite(str(out_dir / fname), bgr)

params = """steps=12000  lr=0.03  lr-min=0.003  init=uniform
alpha=0.01  beta=2.5  batch-size=8
bbox-placement  patch-fraction=1.0  torso-width
iou-loss  iou-sigma=0.5  hard-mining  hard-temp=0.5
geo-warmup=0.20  geo-ramp=0.25
hosts-dir=data/clean (52 images)
best_conf_suppression=-44.8%  (best_obj updated at step ~10540)
suppressed(<0.25)=13/52  at end
patch_best  = checkpoint at best_obj (step ~10540)
patch_final = patch state after exactly 12000 steps
JPEG fix (real cv2 JPEG + STE) applied this run
"""
(out_dir / "params.txt").write_text(params)

print(f"\nIteration folder: {out_dir}")
print("Done.")
