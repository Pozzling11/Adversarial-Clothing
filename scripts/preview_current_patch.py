"""
Quick preview of current patch on 3 sample images.
Loads the latest checkpoint and composites it onto 3 clean images.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load checkpoint
ckpt_path = PROJECT_ROOT / "patterns" / "patch_160_jumpsuit_ckpt.pt"
print(f"[INFO] Loading checkpoint from {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu")

# Extract patch and shape mask
patch = ckpt.get("best_patch", ckpt.get("patch"))  # Use best_patch if available
shape_logits = ckpt.get("best_shape_logits", ckpt.get("shape_logits", None))

print(f"[INFO] Checkpoint step: {ckpt.get('step', '?')}")
print(f"[INFO] Patch shape: {patch.shape}")
print(f"[INFO] Has shape mask: {shape_logits is not None}")

if shape_logits is not None:
    shape_mask = torch.sigmoid(shape_logits) >= 0.5
    mask_coverage = shape_mask.float().mean().item()
    print(f"[INFO] Mask coverage: {mask_coverage:.1%}")

# Convert patch to numpy [0, 255]
patch_np = (patch.clamp(0, 1).numpy() * 255).astype(np.uint8)
if patch_np.shape[0] == 1:
    patch_np = patch_np[0]  # Remove batch dimension if present
if patch_np.shape[0] == 3:
    patch_np = np.transpose(patch_np, (1, 2, 0))
    patch_np = cv2.cvtColor(patch_np, cv2.COLOR_RGB2BGR)
elif patch_np.shape[0] == 4:
    # Has alpha channel
    patch_np = np.transpose(patch_np, (1, 2, 0))
    patch_np = cv2.cvtColor(patch_np[:, :, :3], cv2.COLOR_RGB2BGR)

print(f"[INFO] Patch size: {patch_np.shape}")

# Get 3 clean images
clean_dir = PROJECT_ROOT / "data" / "clean"
image_files = sorted([f for f in clean_dir.glob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".avif"}])[:3]

print(f"[INFO] Found {len(image_files)} sample images")

# Create collage
preview_images = []

for idx, img_path in enumerate(image_files):
    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        # Try PIL for AVIF support
        img_pil = Image.open(img_path).convert("RGB")
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    h, w = img.shape[:2]
    print(f"[INFO] Image {idx+1}: {img_path.name} ({w}×{h})")
    
    # Simple center compositing
    p_h, p_w = patch_np.shape[:2]
    x = max(0, (w - p_w) // 2)
    y = max(0, (h - p_h) // 2)
    
    # Simple alpha blend if patch has alpha
    if patch.shape[0] == 4 or (len(patch.shape) > 1 and patch.shape[0] == 4):
        alpha = patch[-1:].numpy()
        if alpha.shape[0] == 1:
            alpha = alpha[0]
        alpha_np = np.transpose(alpha, (1, 2, 0)) if len(alpha.shape) == 3 else alpha
        img_crop = img[y:min(y+p_h, h), x:min(x+p_w, w)]
        patch_crop = patch_np[0:min(p_h, img_crop.shape[0]), 0:min(p_w, img_crop.shape[1])]
        if img_crop.shape[:2] == patch_crop.shape[:2]:
            img[y:y+img_crop.shape[0], x:x+img_crop.shape[1]] = patch_crop
    else:
        # Direct overlay
        if y + p_h <= h and x + p_w <= w:
            img[y:y+p_h, x:x+p_w] = patch_np
    
    # Resize for display (keep aspect ratio)
    scale = 400 / max(h, w)
    disp_h, disp_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
    
    # Convert to RGB for PIL
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    preview_images.append(Image.fromarray(img_rgb))

# Create horizontal collage
total_w = sum(img.width for img in preview_images) + 20
max_h = max(img.height for img in preview_images)
collage = Image.new("RGB", (total_w + 20, max_h + 20), color=(255, 255, 255))

x_pos = 10
for img in preview_images:
    collage.paste(img, (x_pos, 10))
    x_pos += img.width + 10

output_path = PROJECT_ROOT / "preview_current_patch.png"
collage.save(output_path)
print(f"\n[INFO] Preview saved to {output_path}")
collage.show()

# Also save just the patch
patch_output = PROJECT_ROOT / "preview_current_patch_only.png"
Image.fromarray(cv2.cvtColor(patch_np, cv2.COLOR_BGR2RGB)).save(patch_output)
print(f"[INFO] Patch-only saved to {patch_output}")
