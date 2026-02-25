# Physical-World Adversarial Robustness — YOLOv8n Red-Team Experiment

> **Research context:** Red-team exercise to evaluate the susceptibility of YOLOv8n to
> physical-world adversarial patches (false-negative / evasion attacks) and to inform
> defensive hardening strategies.  All testing is conducted in an authorised,
> controlled environment.

---

## Project layout

```
.
├── main.py                  # Live webcam detection viewer (baseline)
├── yolov8n.pt               # Pre-downloaded local weights
├── requirements.txt
│
├── scripts/
│   ├── generate_patch.py    # Step 1 – PGD-based 128×128 adversarial patch synthesis
│   ├── apply_patch.py       # Step 2 – EOT compositing onto clean images
│   └── evaluate.py          # Step 3 – Detection audit + CSV logging
│
├── data/
│   ├── clean/               # ← place your baseline test images here
│   ├── adversarial/         # auto-populated by apply_patch.py
│   └── physical/            # photos of real-world print tests
│
├── patterns/                # synthesised patch PNGs land here
└── results/                 # timestamped CSV reports from evaluate.py
```

---

## Requirements

- Python 3.10+
- A webcam (or provide another video source)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Experiment workflow

### Step 1 — Generate the adversarial patch

Uses PGD (Projected Gradient Descent) to optimise a 128×128 RGB patch that
minimises the `person`-class confidence scores in YOLOv8n's raw output.

```bash
python scripts/generate_patch.py
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--steps` | `500` | PGD optimisation iterations |
| `--lr` | `0.01` | PGD step size |
| `--host` | `None` | Background image; blank canvas used if omitted |
| `--out` | `patterns/patch_128.png` | Output path |
| `--verbose` | off | Print loss every 50 steps |

The script prints a baseline confidence and a final confidence and saves:
- `patterns/patch_128.png` — the optimised patch
- `patterns/patch_128_preview.png` — 4× upscaled view for visual inspection

---

### Step 2 — Apply patch with EOT transforms

Composites the patch onto every image in `data/clean/` using the
Expectation-over-Transformation (EOT) stack:

| Transform | Range |
|-----------|-------|
| Rescale | ±30 % |
| Rotation | ±15 ° |
| Perspective warp | subtle |
| Gaussian blur | 0–3 px |
| Brightness/contrast jitter | ±30 % / ±20 |

```bash
python scripts/apply_patch.py
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--placement` | `center` | `center`, `random`, `top-left`, `top-right` |
| `--n-aug` | `1` | Augmented copies per clean image |
| `--scale-range` | `0.7 1.3` | EOT scale range |
| `--rot-range` | `15.0` | Max rotation in degrees |

Output images land in `data/adversarial/` with matching filenames.

---

### Step 3 — Evaluate and log to CSV

Runs YOLOv8n on both sets, records every detection, and writes a timestamped
CSV to `results/`.

```bash
python scripts/evaluate.py
```

CSV columns:

| Column | Description |
|--------|-------------|
| `image` | Filename |
| `set` | `clean` or `adversarial` |
| `person_detected` | `True` / `False` |
| `max_person_conf` | Highest person confidence score (0.0 if none) |
| `num_person_dets` | Number of person bounding boxes |
| `is_false_negative` | `True` = attack success (evasion) |

The evaluator also prints a summary table showing detection rate, average
confidence, and false-negative rate.

---

## Live detection viewer (baseline)

```bash
python main.py
```

Press `q` to quit.

Optional arguments:

```bash
python main.py --model yolov8s.pt --source 0 --conf 0.3 --iou 0.45 --imgsz 640
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `yolov8n.pt` | Weights file |
| `--source` | `0` | Camera index or file/stream path |
| `--conf` | `0.25` | Confidence threshold |
| `--iou` | `0.45` | NMS IoU threshold |
| `--imgsz` | `640` | Inference resolution |

---

## Physical transfer testing

1. Print `patterns/patch_128_preview.png` at the target size (e.g. A4 / fabric).
2. Photograph the printed patch applied to a person at target distances and angles.
3. Copy photos to `data/physical/` (keep `data/clean/` originals for pairing).
4. Re-run `evaluate.py --clean-dir data/clean --adv-dir data/physical` to measure
   the **transfer gap** = (digital FN rate) − (physical FN rate).

---

## Defensive hardening (next steps)

- **Input randomisation** — random resize + pad before inference
- **Temporal voting** — require N consecutive frames before confirming a detection
- **Adversarial fine-tuning** — augment training data with synthesised patches
- **Confidence calibration** — temperature scaling to tighten threshold sensitivity
