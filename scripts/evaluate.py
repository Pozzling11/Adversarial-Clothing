"""
evaluate.py
===========
Evaluation harness for the adversarial patch experiment.

Runs YOLOv8n on images in two sets (clean and adversarial), logs every per-image
detection result to a timestamped CSV in results/, and prints a summary table.

CSV columns
-----------
  image          : filename (basename)
  set            : 'clean' or 'adversarial'
  person_detected: True / False  (any detection above --conf threshold)
  max_person_conf: highest person confidence across all detections (0.0 if none)
  num_person_dets: total person bounding boxes in this frame
  is_false_negative: True when set=='adversarial' AND paired clean image had
                     person_detected==True but this image has person_detected==False

A 'success' for the attack is an is_false_negative == True.

Usage
-----
  python scripts/evaluate.py
  python scripts/evaluate.py --clean-dir data/clean --adv-dir data/adversarial
  python scripts/evaluate.py --conf 0.25 --model yolov8n.pt
"""

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

PROJECT_ROOT    = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = str(PROJECT_ROOT / "yolov8n.pt")
DEFAULT_CLEAN   = str(PROJECT_ROOT / "data" / "clean")
DEFAULT_ADV     = str(PROJECT_ROOT / "data" / "adversarial")
DEFAULT_RESULTS = str(PROJECT_ROOT / "results")
SUPPORTED_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
PERSON_CLASS_ID = 0   # COCO class 0 = 'person'


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adversarial patch evaluator – logs to CSV")
    p.add_argument("--model",      default=DEFAULT_WEIGHTS, help="YOLOv8n .pt weights path")
    p.add_argument("--clean-dir",  default=DEFAULT_CLEAN,   help="Clean images directory")
    p.add_argument("--adv-dir",    default=DEFAULT_ADV,     help="Adversarial images directory")
    p.add_argument("--results",    default=DEFAULT_RESULTS, help="Directory to write result CSV")
    p.add_argument("--conf",  type=float, default=0.25,     help="Detection confidence threshold (default 0.25)")
    p.add_argument("--iou",   type=float, default=0.45,     help="NMS IoU threshold (default 0.45)")
    p.add_argument("--imgsz", type=int,   default=640,      help="Inference image size (default 640)")
    p.add_argument("--noise-type",  default=None,
                   help="Noise/init type label written to every CSV row "
                        "(auto-detected from --adv-dir path if omitted)")
    p.add_argument("--save-annotated", action="store_true",
                   help="Write annotated preview images alongside the CSV")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Detection helper
# ---------------------------------------------------------------------------

def detect_persons(
    model: YOLO,
    img: "np.ndarray",        # type: ignore[name-defined]  # noqa: F821
    conf: float,
    iou: float,
    imgsz: int,
) -> tuple[bool, float, int]:
    """
    Returns
    -------
    person_detected : bool    – at least one person above threshold
    max_conf        : float   – highest person-class confidence (0.0 if none)
    num_dets        : int     – total number of person boxes
    """
    results = model.predict(
        source=img,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
        classes=[PERSON_CLASS_ID],   # restrict to 'person' only for efficiency
    )
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return False, 0.0, 0

    confidences = [
        float(box.conf.item())
        for box in result.boxes
        if int(box.cls.item()) == PERSON_CLASS_ID
    ]

    if not confidences:
        return False, 0.0, 0

    return True, max(confidences), len(confidences)


# ---------------------------------------------------------------------------
# Directory scan
# ---------------------------------------------------------------------------

def load_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTS and not p.name.startswith("._")
    )


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "image",
    "set",
    "noise_type",
    "person_detected",
    "max_person_conf",
    "num_person_dets",
    "is_false_negative",
]


def write_csv(rows: list[dict], out_path: Path) -> None:
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(rows: list[dict]) -> None:
    clean_rows = [r for r in rows if r["set"] == "clean"]
    adv_rows   = [r for r in rows if r["set"] == "adversarial"]

    # Collect unique noise types present in the adv set
    noise_types = sorted({r["noise_type"] for r in adv_rows if r["noise_type"] != "clean"})
    noise_label = ", ".join(noise_types) if noise_types else "unknown"

    def avg_conf(subset):
        vals = [r["max_person_conf"] for r in subset if r["person_detected"]]
        return sum(vals) / len(vals) if vals else 0.0

    def detection_rate(subset):
        if not subset:
            return 0.0
        return sum(1 for r in subset if r["person_detected"]) / len(subset) * 100

    fn_count = sum(1 for r in adv_rows if r["is_false_negative"])
    fn_rate  = fn_count / len(adv_rows) * 100 if adv_rows else 0.0

    print()
    print("=" * 56)
    print("  EXPERIMENT SUMMARY")
    print("=" * 56)
    print(f"  Noise type(s)            : {noise_label}")
    print(f"  Clean images evaluated   : {len(clean_rows)}")
    print(f"  Adversarial imgs eval    : {len(adv_rows)}")
    print()
    print(f"  Clean detection rate     : {detection_rate(clean_rows):>6.1f}%")
    print(f"  Adv   detection rate     : {detection_rate(adv_rows):>6.1f}%")
    print()
    print(f"  Clean avg person conf    : {avg_conf(clean_rows):>6.4f}")
    print(f"  Adv   avg person conf    : {avg_conf(adv_rows):>6.4f}")
    conf_drop = (avg_conf(clean_rows) - avg_conf(adv_rows)) / max(avg_conf(clean_rows), 1e-9) * 100
    print(f"  Confidence drop          : {conf_drop:>6.1f}%")
    print()
    print(f"  False Negatives (attack successes) : {fn_count} / {len(adv_rows)}")
    print(f"  False Negative rate      : {fn_rate:>6.1f}%")
    print("=" * 56)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)

    clean_dir = Path(args.clean_dir)
    adv_dir   = Path(args.adv_dir)

    # Auto-detect noise type from --adv-dir last path component or explicit flag
    if args.noise_type:
        noise_type = args.noise_type
    else:
        # e.g. data/adversarial/blocky  →  "blocky"
        #       data/adversarial         →  "unknown"
        candidate = adv_dir.name
        noise_type = candidate if candidate not in ("adversarial", "") else "unknown"

    print(f"[INFO] Noise type : {noise_type}")

    clean_images = load_images(clean_dir)
    adv_images   = load_images(adv_dir)

    if not clean_images and not adv_images:
        print("[WARN] No images found in either clean or adversarial directories.")
        print("  → Populate data/clean/ and run apply_patch.py before evaluating.")
        return

    print(f"[INFO] Model      : {args.model}")
    print(f"[INFO] Conf thresh: {args.conf}  |  IoU: {args.iou}  |  imgsz: {args.imgsz}")
    print(f"[INFO] Clean imgs : {len(clean_images)}")
    print(f"[INFO] Adv   imgs : {len(adv_images)}")

    model = YOLO(args.model)

    rows: list[dict] = []

    # ------------------------------------------------------------------
    # Process CLEAN set
    # ------------------------------------------------------------------
    print(f"\n[INFO] Evaluating clean set …")
    clean_detections: dict[str, bool] = {}

    for img_path in clean_images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Skipping unreadable: {img_path.name}")
            continue

        detected, max_conf, num_dets = detect_persons(model, img, args.conf, args.iou, args.imgsz)
        clean_detections[img_path.name] = detected

        rows.append({
            "image":            img_path.name,
            "set":              "clean",
            "noise_type":       "clean",
            "person_detected":  detected,
            "max_person_conf":  round(max_conf, 6),
            "num_person_dets":  num_dets,
            "is_false_negative": False,
        })

        status = "DETECTED" if detected else "NOT DETECTED"
        print(f"  {img_path.name:<40}  {status}  conf={max_conf:.4f}  dets={num_dets}")

    # ------------------------------------------------------------------
    # Process ADVERSARIAL set
    # ------------------------------------------------------------------
    print(f"\n[INFO] Evaluating adversarial set …")

    for img_path in adv_images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Skipping unreadable: {img_path.name}")
            continue

        detected, max_conf, num_dets = detect_persons(model, img, args.conf, args.iou, args.imgsz)

        # Determine paired clean image name:
        # Handles both exact match (n_aug=1) and aug suffix (e.g. img_aug00.jpg → img.jpg)
        stem = img_path.stem
        ext  = img_path.suffix
        pair_name = img_path.name
        if "_aug" in stem:
            base_stem = stem.rsplit("_aug", 1)[0]
            pair_name = base_stem + ext

        clean_had_person = clean_detections.get(pair_name, True)  # default True = assume clean had person
        is_fn = clean_had_person and (not detected)

        rows.append({
            "image":             img_path.name,
            "set":               "adversarial",
            "noise_type":        noise_type,
            "person_detected":   detected,
            "max_person_conf":   round(max_conf, 6),
            "num_person_dets":   num_dets,
            "is_false_negative": is_fn,
        })

        status = "DETECTED" if detected else "NOT DETECTED"
        fn_tag = "  ← FALSE NEGATIVE  ✓ ATTACK SUCCESS" if is_fn else ""
        print(f"  {img_path.name:<40}  {status}  conf={max_conf:.4f}  dets={num_dets}{fn_tag}")

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path  = results_dir / f"detections_{timestamp}.csv"
    write_csv(rows, csv_path)
    print(f"\n[INFO] Results saved → {csv_path}")

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print_summary(rows)


if __name__ == "__main__":
    main()
