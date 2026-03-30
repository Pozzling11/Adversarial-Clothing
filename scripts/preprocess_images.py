"""
preprocess_images.py
-------------------
Script to preprocess all images in a directory:
- Downscale images larger than a target size (e.g., 1280x960)
- Optionally skip or log corrupted/unreadable images
- Save processed images to a new directory, preserving filenames

Usage:
  python scripts/preprocess_images.py --input-dir data/TRAINING\ LEG\ IMAGES --output-dir data/TRAINING\ LEG\ IMAGES\ _preprocessed --max-width 1280 --max-height 960
"""
import os
import cv2
import argparse
from pathlib import Path

def process_image(in_path, out_path, max_width, max_height):
    img = cv2.imread(str(in_path))
    if img is None:
        print(f"[WARN] Could not read: {in_path}")
        return False
    h, w = img.shape[:2]
    if w > max_width or h > max_height:
        scale = min(max_width / w, max_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"[INFO] Resized {in_path.name}: {w}x{h} -> {new_w}x{new_h}")
    cv2.imwrite(str(out_path), img)
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-width', type=int, default=1280)
    parser.add_argument('--max-height', type=int, default=960)
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    for fname in sorted(os.listdir(in_dir)):
        if not any(fname.lower().endswith(ext) for ext in exts):
            continue
        in_path = in_dir / fname
        out_path = out_dir / fname
        process_image(in_path, out_path, args.max_width, args.max_height)

if __name__ == '__main__':
    main()
