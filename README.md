# Physical-World Adversarial Robustness — YOLOv8n Red-Team Experiment

> **Research context:** Red-team exercise to evaluate the susceptibility of YOLOv8n
> person detection to physical-world adversarial patches (false-negative / evasion
> attacks) and to inform defensive hardening strategies. All testing is conducted in
> an authorised, controlled environment.

---

## Table of contents

- [Overview](#overview)
- [Project layout](#project-layout)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Core workflow](#core-workflow)
  - [Step 1 — Generate the adversarial patch](#step-1--generate-the-adversarial-patch)
  - [Step 2 — Apply patch to images](#step-2--apply-patch-to-images)
  - [Step 3 — Evaluate detection performance](#step-3--evaluate-detection-performance)
- [Advanced training features](#advanced-training-features)
  - [Loss functions](#loss-functions)
  - [Placement strategies](#placement-strategies)
  - [EOT augmentation stack](#eot-augmentation-stack)
  - [Geometric curriculum](#geometric-curriculum)
  - [Hard-example mining](#hard-example-mining)
  - [Checkpointing and resumption](#checkpointing-and-resumption)
- [Live detection viewer](#live-detection-viewer)
- [Utility scripts](#utility-scripts)
- [Video evaluation](#video-evaluation)
- [Physical transfer testing](#physical-transfer-testing)
- [Defensive hardening](#defensive-hardening-next-steps)

---

## Overview

This project implements a full pipeline for synthesising, applying, and evaluating
**adversarial patches** targeting YOLOv8n's person detector. The attack is optimised
for **physical-world transfer**: patches are designed to be printed on clothing (torso
quilts) and survive real-world imaging conditions (lighting changes, camera angles,
motion blur, cloth wrinkles, etc.).

Key capabilities:

- **PGD-based patch optimisation** with differentiable Expectation-over-Transformation (EOT)
- **Pose-guided torso-quilt placement** using YOLOv8n-pose keypoints
- **Multi-host training** with hard-example mining for generalisation
- **Geometric curriculum learning** to stabilise early training
- **Printability and smoothness constraints** (NPS + Total Variation losses)
- **Optional dual-patch** (torso + legs) and **hat patch** modes
- **IoU-weighted anchor suppression** focusing gradient on person-overlapping anchors
- **Physical transfer evaluation** for digital-to-print gap measurement
- Runs on **Apple Silicon (MPS)**, CUDA, or CPU

---

## Project layout

```
.
├── main.py                       # Live webcam detection viewer (baseline)
├── _diag_rawpred.py              # Quick diagnostic: inspect raw YOLOv8n output layout
├── yolov8n.pt                    # YOLOv8n detection weights
├── yolov8n-pose.pt               # YOLOv8n pose estimation weights
├── yolov8n-seg.pt                # YOLOv8n segmentation weights
├── requirements.txt
│
├── scripts/
│   ├── generate_patch.py         # Core trainer — PGD + EOT patch synthesis
│   ├── apply_patch.py            # EOT compositing onto clean images
│   ├── evaluate.py               # Batch detection audit + CSV logging
│   ├── evaluate_video.py         # Per-frame video comparison evaluation
│   ├── compare_inits.py          # Benchmark all init methods head-to-head
│   ├── make_a3_print.py          # Upscale patch to A3 @ 300 DPI for printing
│   ├── preview_current_patch.py  # Quick checkpoint preview on sample images
│   ├── preview_torso_placement.py# Visualise bbox-guided torso patch region
│   ├── preprocess_images.py      # Batch image downscaling utility
│   ├── saliency_utils.py         # Gradient-based saliency map computation
│   ├── template_crop_utils.py    # Body silhouette template cropping helpers
│   ├── visualize_template_placement.py  # Overlay template on segmentation masks
│   ├── visualize_leg_patch.py    # Keypoint + leg region overlay (4 images)
│   ├── visualize_leg_patch_extra.py     # Extended leg visualisation
│   ├── visualize_leg_pose.py     # Full keypoint labelling (10 images)
│   ├── visualize_legs_in_training_images.py  # Leg viz on training data
│   └── visualize_dilation.py     # Mask dilation diagnostics
│
├── data/
│   ├── clean/                    # Baseline test images (place yours here)
│   ├── adversarial/              # Auto-populated by apply_patch.py
│   ├── physical/                 # Photos of real-world print tests
│   ├── printable_colors.txt      # Printable colour palette for NPS loss
│   ├── TRAINING LEG IMAGES/      # Raw leg-focused training images
│   └── TRAINING LEG IMAGES _preprocessed/  # Preprocessed versions
│
├── patterns/
│   ├── iterations/               # Patch snapshots per training iteration
│   ├── targets/                  # Target templates (e.g. smiley.png)
│   └── *_ckpt.pt                 # Training checkpoints (step, best_patch, optimizer)
│
├── results/                      # Timestamped CSV detection logs
│
├── potential shapes/             # Patch shape exploration & previews
├── leg_pose_vis/                 # Pose keypoint visualisation outputs
├── torso_quilt_vis/              # Torso quilt placement debug outputs
├── torso_width_preview/          # Torso region preview outputs
├── debug_leg_patch_overlays/     # Leg patch debug overlays
└── template_placement_vis/       # Template alignment diagnostics
```

---

## Requirements

- **Python 3.10+**
- **PyTorch** with MPS (Apple Silicon), CUDA, or CPU backend
- A webcam for the live viewer (optional)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies: `ultralytics ≥ 8.3.0`, `opencv-python ≥ 4.8.0` (PyTorch is pulled
in transitively by ultralytics).

> **Apple Silicon note:** Set `PYTORCH_ENABLE_MPS_FALLBACK=1` when running the
> trainer to allow PyTorch to fall back to CPU for unsupported MPS operations.

---

## Quick start

```bash
# 1. Train a torso-quilt patch (160px, 15k steps)
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/generate_patch.py \
  --patch-size 160 --steps 15000 --lr 0.02 --lr-min 0.002 \
  --init uniform --alpha 0.01 --beta 2.5 --batch-size 8 \
  --bbox-placement --torso-width --patch-fraction 1.0 \
  --iou-loss --iou-sigma 0.5 \
  --hard-mining --hard-temp 0.5 \
  --geo-warmup 0.15 --geo-ramp 0.20 \
  --checkpoint-every 100 --resume \
  --hosts-dir data/clean \
  --out patterns/patch_160_torso_quilt.png

# 2. Apply patch to test images
python scripts/apply_patch.py \
  --patch patterns/patch_160_torso_quilt.png \
  --placement torso

# 3. Evaluate detection performance
python scripts/evaluate.py

# 4. Prepare for physical printing
python scripts/make_a3_print.py patterns/iterations/iteration_30/patch.png
```

---

## Core workflow

### Step 1 — Generate the adversarial patch

Uses **Projected Gradient Descent (PGD)** to optimise an RGB patch that minimises
person-class confidence scores in YOLOv8n's output. Training runs over a pool of
host images with differentiable EOT augmentation for real-world robustness.

```bash
python scripts/generate_patch.py [options]
```

**Key arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--patch-size` | `256` | Patch resolution (square, in pixels) |
| `--steps` | `1500` | Number of PGD optimisation iterations |
| `--lr` | `0.01` | Initial learning rate |
| `--lr-min` | `0.001` | Minimum LR (cosine decay schedule) |
| `--batch-size` | `16` | Host images sampled per PGD step |
| `--hosts-dir` | — | Directory of background host images |
| `--init` | `uniform` | Initialisation: `uniform`, `gaussian`, `checkerboard`, `stripes`, `salt_pepper`, `gray`, `blocky`, `perlin` |
| `--alpha` | `0.01` | NPS (printability) loss weight |
| `--beta` | `2.5` | Total Variation (smoothness) loss weight |
| `--out` | `patterns/patch_256.png` | Output patch path |
| `--seed` | — | Random seed for reproducibility |
| `--verbose` | off | Verbose logging |
| `--log-file` | — | Write training log to file |
| `--no-eot` | off | Disable all EOT augmentation |

The trainer saves the optimised patch to `--out` and a 4× upscaled preview alongside it.

---

### Step 2 — Apply patch to images

Composites the patch onto every image in `data/clean/` using the EOT transform stack.

```bash
python scripts/apply_patch.py [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--patch` | — | Path to the adversarial patch PNG |
| `--placement` | `center` | `center`, `random`, `torso`, `top-left`, `top-right` |
| `--display-size` | `160` | Resize patch before compositing |
| `--n-aug` | `1` | Augmented copies per clean image |
| `--scale-range` | `0.7 1.3` | EOT scale variation bounds |

Output images are written to `data/adversarial/` with matching filenames.

---

### Step 3 — Evaluate detection performance

Runs YOLOv8n on both clean and adversarial image sets, logs per-image stats to a
timestamped CSV in `results/`.

```bash
python scripts/evaluate.py [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--clean-dir` | `data/clean` | Clean image directory |
| `--adv-dir` | `data/adversarial` | Adversarial image directory |
| `--conf` | `0.25` | Detection confidence threshold |
| `--iou` | `0.45` | NMS IoU threshold |
| `--imgsz` | `640` | Inference image size |

**CSV columns:**

| Column | Description |
|--------|-------------|
| `image` | Filename |
| `set` | `clean` or `adversarial` |
| `person_detected` | `True` / `False` |
| `max_person_conf` | Highest person confidence (0.0 if none) |
| `num_person_dets` | Number of person bounding boxes |
| `is_false_negative` | `True` = successful evasion |

A summary table is printed with detection rate, average confidence, and false-negative rate.

---

## Advanced training features

### Loss functions

The total loss combines multiple objectives:

$$L = L_{\text{adv}} + \alpha \cdot L_{\text{nps}} + \beta \cdot L_{\text{tv}} \; [+ \lambda_{\text{attn}} \cdot L_{\text{attn}} + \lambda_{\text{letter}} \cdot L_{\text{letter}}]$$

| Loss | Description |
|------|-------------|
| **Adversarial** ($L_{\text{adv}}$) | Mean of top-k person confidence scores + max-term for worst-case suppression |
| **NPS** ($L_{\text{nps}}$) | Non-Printability Score — penalises colours far from a printable palette (`data/printable_colors.txt`) |
| **Total Variation** ($L_{\text{tv}}$) | Encourages spatial smoothness: $\sum_{i,j} [(p_{i,j} - p_{i+1,j})^2 + (p_{i,j} - p_{i,j+1})^2]$ |
| **Attention** ($L_{\text{attn}}$) | Optional saliency-weighted contrast to redirect model attention (experimental) |
| **Letter shape** ($L_{\text{letter}}$) | Optional MSE between patch luminance and a rendered character mask |

### Placement strategies

| Strategy | Flag | Description |
|----------|------|-------------|
| **Torso quilt** | `--bbox-placement --torso-width` | Tiles the patch across the shoulder → hip region using YOLOv8n-pose keypoints. Shoulder width determines tile size. Random x/y offsets per step simulate quilt misalignment. |
| **Bbox-guided** | `--bbox-placement` | Scales patch relative to detected person bbox height |
| **Hat patch** | `--hat-patch` | Adds a secondary patch on the head region (crown area) |
| **Dual patch** | `--dual-patch` | Trains torso + leg patches jointly |
| **IoU-weighted** | `--iou-loss --iou-sigma σ` | Weights anchor contributions by Gaussian proximity to the detected person bbox, focusing gradient on person-overlapping anchors |

### EOT augmentation stack

All transforms are **differentiable** and applied per training step to ensure the
learned patch is robust to real-world imaging conditions:

| Transform | Range / Details |
|-----------|-----------------|
| Brightness / contrast jitter | ±30% / ±40 offset |
| Gamma correction | 0.6 – 1.8 |
| Per-channel colour shift | ±20% |
| Gaussian blur | kernel 1, 3, or 5 |
| Trimodal scale | 60% close (0.50–1.20×), 25% medium (0.35–0.55×), 15% far (0.15–0.35×) |
| Rotation | ±20° via horizontal shear approximation |
| Perspective warp | ±25% fractional jitter per corner |
| Random shadow strips | 40% probability |
| JPEG compression | Quality 55–95 (straight-through estimator for gradient flow) |
| Print grain noise | Gaussian, std 0–8 |
| Cloth wrinkle deformation | 50% probability, subtle fold displacement |
| Post-spatial jitter | ±10% relative to patch region |

### Geometric curriculum

To stabilise early training (especially on MPS), geometric augmentation is introduced
gradually:

| Phase | Steps | Augmentation |
|-------|-------|-------------|
| Warmup | 0 → `geo-warmup × total` | Photometric only (no rotation, perspective, scale) |
| Ramp | warmup → `(warmup + geo-ramp) × total` | Geometric probability linearly ramps 0 → 100% |
| Full | remaining steps | Full EOT (photometric + geometric) |

Controlled via `--geo-warmup` and `--geo-ramp` (fractions of total steps).

### Hard-example mining

When `--hard-mining` is enabled, host images are **sampled proportional to their
recent loss** (EMA-tracked per image). This implements a curriculum that focuses
training on images where the patch is least effective.

- `--hard-temp` controls the softmax temperature (lower = more aggressive mining)

### Checkpointing and resumption

- `--checkpoint-every N` saves `{out_stem}_ckpt.pt` every N steps
- `--resume` loads the latest checkpoint and continues training
- Checkpoint contains: `step`, `best_patch`, `best_loss`, `optimizer_state`, `patch_tensor`
- A fallback system uses batch loss until the first confidence evaluation, then switches to eval-based best-patch selection (every 200 steps)

---

## Live detection viewer

```bash
python main.py
```

Opens a live webcam feed with YOLOv8n detection overlays. Press `q` to quit.

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `yolov8n.pt` | Weights file |
| `--source` | `0` | Camera index or file/stream path |
| `--conf` | `0.25` | Confidence threshold |
| `--iou` | `0.45` | NMS IoU threshold |
| `--imgsz` | `640` | Inference resolution |

---

## Utility scripts

| Script | Purpose |
|--------|---------|
| `compare_inits.py` | Train one patch per initialisation method, evaluate all, rank by evasion success rate |
| `make_a3_print.py` | Upscale a patch to A3 size (3508 × 4961 px @ 300 DPI) for physical printing |
| `preview_current_patch.py` | Load a checkpoint and composite the best patch onto sample images for quick sanity checks |
| `preview_torso_placement.py` | Visualise the bbox-guided torso region overlay on host images |
| `preprocess_images.py` | Batch downscale images to a target resolution (`--max-width`, `--max-height`) |
| `visualize_template_placement.py` | Overlay body silhouette template on YOLOv8n-seg masks |
| `visualize_leg_patch.py` | Draw COCO keypoints + leg region annotation on 4 random images |
| `visualize_leg_pose.py` | Full keypoint labelling on 10 images |
| `visualize_dilation.py` | Mask dilation diagnostic visualisations |
| `saliency_utils.py` | Gradient-based saliency map computation (library, not standalone) |
| `template_crop_utils.py` | Body template cropping helpers (library, not standalone) |
| `_diag_rawpred.py` | Inspect raw YOLOv8n output shape and top-k person confidence for debugging |

---

## Video evaluation

Compare detection performance on paired clean/adversarial videos:

```bash
python scripts/evaluate_video.py --clean video_clean.mp4 --adv video_adv.mp4 \
  --save-video --output-dir results/
```

Outputs per-frame false-negative rate, average confidence, and an optional
side-by-side annotated video.

---

## Physical transfer testing

1. Generate a patch using the training pipeline.
2. Upscale for printing: `python scripts/make_a3_print.py patterns/patch.png`
3. Print the pattern on fabric or paper at the target size.
4. Photograph the printed patch worn by a person at various distances and angles.
5. Copy photos to `data/physical/`.
6. Evaluate the physical effectiveness:
   ```bash
   python scripts/evaluate.py --clean-dir data/clean --adv-dir data/physical
   ```
7. Measure the **transfer gap** = (digital FN rate) − (physical FN rate).

---

## Defensive hardening (next steps)

- **Input randomisation** — random resize + pad before inference
- **Temporal voting** — require N consecutive frames before confirming a detection
- **Adversarial fine-tuning** — augment training data with synthesised patches
- **Confidence calibration** — temperature scaling to tighten threshold sensitivity

---

## Iteration history

Over 30 training iterations have been conducted, exploring different configurations:

| Iteration range | Focus |
|-----------------|-------|
| 0–9 | Initial patch synthesis and baseline experiments |
| 10–11 | Hat + torso dual-patch exploration |
| 12–14 | Torso sizing (25–35% of bbox), harsh lighting robustness |
| 15–21 | EOT refinement, curriculum tuning, hard mining |
| 22–30 | Full pipeline with IoU loss, quilt tiling, shape learning |

Iteration snapshots are stored in `patterns/iterations/` for reproducibility.
