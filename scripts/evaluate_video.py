"""
evaluate_video.py — Quantitative comparison of adversarial patch effect on video.

Usage:
    python scripts/evaluate_video.py --clean data/videos/clean.mp4 --adv data/videos/adv.mp4

Outputs:
    - Console summary table (FN rate, avg conf, conf drop)
    - CSV saved to results/video_evaluation_<timestamp>.csv
    - Optionally side-by-side annotated MP4 (--save-video)
"""

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO


# ─── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate adversarial patch effect across two video files."
    )
    parser.add_argument("--clean",  required=True,  help="Path to clean video (no patch)")
    parser.add_argument("--adv",    required=True,  help="Path to adversarial video (patch present)")
    parser.add_argument("--model",  default="yolov8n.pt", help="YOLOv8 weights (default: yolov8n.pt)")
    parser.add_argument("--conf",   type=float, default=0.25, help="Detection confidence threshold (default: 0.25)")
    parser.add_argument("--iou",    type=float, default=0.45, help="NMS IoU threshold (default: 0.45)")
    parser.add_argument("--imgsz",  type=int,   default=640,  help="Inference image size (default: 640)")
    parser.add_argument("--save-video", action="store_true",
                        help="Save annotated output video to results/")
    parser.add_argument("--output-dir", default="results", help="Directory for CSV and video output")
    return parser.parse_args()


# ─── HELPERS ────────────────────────────────────────────────────────────────────

PERSON_CLASS_ID = 0  # COCO class 0 = person


def process_video(
    path: str,
    model: YOLO,
    conf_thresh: float,
    iou: float,
    imgsz: int,
    label: str,
    save_video: bool,
    output_dir: Path,
) -> tuple[list[dict], cv2.VideoWriter | None]:
    """Run YOLOv8 on every frame and return per-frame stats."""

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {path}", file=sys.stderr)
        sys.exit(1)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if save_video:
        out_path = output_dir / f"annotated_{label}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    records: list[dict] = []
    frame_idx = 0

    print(f"\nProcessing [{label}] — {total} frames …")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1

        results = model.predict(
            source=frame,
            conf=conf_thresh,
            iou=iou,
            imgsz=imgsz,
            verbose=False,
            classes=[PERSON_CLASS_ID],   # only detect person
        )

        result  = results[0]
        boxes   = result.boxes

        # Collect all person-class detections
        person_confs = []
        if boxes is not None:
            for box in boxes:
                if int(box.cls.item()) == PERSON_CLASS_ID:
                    person_confs.append(float(box.conf.item()))

        detected   = len(person_confs) > 0
        max_conf   = max(person_confs) if detected else 0.0

        records.append({
            "source":    label,
            "frame":     frame_idx,
            "detected":  int(detected),
            "n_persons": len(person_confs),
            "max_conf":  round(max_conf, 4),
        })

        # Progress
        if frame_idx % 30 == 0 or frame_idx == total:
            pct = 100 * frame_idx / max(total, 1)
            print(f"  {frame_idx}/{total} ({pct:.0f}%)  detected={detected}  conf={max_conf:.3f}")

        # Annotate frame
        if writer is not None:
            annotated = result.plot()
            status_color = (0, 200, 0) if detected else (0, 0, 220)
            status_text  = f"DETECTED  conf={max_conf:.2f}" if detected else "NOT DETECTED"
            cv2.putText(annotated, status_text, (12, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2, cv2.LINE_AA)
            writer.write(annotated)

    cap.release()
    if writer:
        writer.release()
        print(f"  Saved annotated video → {output_dir / ('annotated_' + label + '.mp4')}")

    return records


# ─── STATS ──────────────────────────────────────────────────────────────────────

def summarise(records: list[dict]) -> dict:
    total      = len(records)
    detected   = sum(r["detected"] for r in records)
    fn_count   = total - detected
    fn_rate    = 100.0 * fn_count / max(total, 1)

    # Average conf only over frames where person was detected
    det_confs  = [r["max_conf"] for r in records if r["detected"]]
    avg_conf   = sum(det_confs) / len(det_confs) if det_confs else 0.0

    return {
        "total_frames":    total,
        "detected_frames": detected,
        "fn_frames":       fn_count,
        "fn_rate_pct":     round(fn_rate, 2),
        "avg_conf":        round(avg_conf, 4),
    }


# ─── MAIN ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # Process both videos
    clean_records = process_video(
        args.clean, model, args.conf, args.iou, args.imgsz,
        "clean", args.save_video, output_dir,
    )
    adv_records = process_video(
        args.adv, model, args.conf, args.iou, args.imgsz,
        "adversarial", args.save_video, output_dir,
    )

    clean_stats = summarise(clean_records)
    adv_stats   = summarise(adv_records)

    # Derived metrics
    conf_drop    = clean_stats["avg_conf"] - adv_stats["avg_conf"]
    fn_rate_gain = adv_stats["fn_rate_pct"] - clean_stats["fn_rate_pct"]

    # ── Print summary table ──────────────────────────────────────────────────
    print("\n" + "═" * 58)
    print(f"{'METRIC':<30}  {'CLEAN':>10}  {'ADVERSARIAL':>12}")
    print("─" * 58)
    print(f"{'Total frames':<30}  {clean_stats['total_frames']:>10}  {adv_stats['total_frames']:>12}")
    print(f"{'Detected frames':<30}  {clean_stats['detected_frames']:>10}  {adv_stats['detected_frames']:>12}")
    print(f"{'False-negative frames':<30}  {clean_stats['fn_frames']:>10}  {adv_stats['fn_frames']:>12}")
    print(f"{'FN rate (%)':<30}  {clean_stats['fn_rate_pct']:>9.1f}%  {adv_stats['fn_rate_pct']:>11.1f}%")
    print(f"{'Avg confidence (detected)':<30}  {clean_stats['avg_conf']:>10.4f}  {adv_stats['avg_conf']:>12.4f}")
    print("─" * 58)
    print(f"{'Confidence drop':<30}  {conf_drop:>+10.4f}")
    print(f"{'FN rate increase':<30}  {fn_rate_gain:>+9.1f}%")
    print("═" * 58)

    # ── Save CSV ─────────────────────────────────────────────────────────────
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path   = output_dir / f"video_evaluation_{timestamp}.csv"
    all_records = clean_records + adv_records

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "frame", "detected", "n_persons", "max_conf"])
        writer.writeheader()
        writer.writerows(all_records)

    # Summary row appended as comments
    with open(csv_path, "a") as f:
        f.write(f"\n# clean_fn_rate,{clean_stats['fn_rate_pct']}\n")
        f.write(f"# adv_fn_rate,{adv_stats['fn_rate_pct']}\n")
        f.write(f"# conf_drop,{round(conf_drop, 4)}\n")
        f.write(f"# fn_rate_gain,{round(fn_rate_gain, 2)}\n")

    print(f"\nCSV saved → {csv_path}")


if __name__ == "__main__":
    main()
