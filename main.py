import argparse
from typing import Dict, Tuple

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live YOLOv8 object detection on webcam/video stream")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 model weights")
    parser.add_argument("--source", default="0", help="Camera index (e.g. 0) or video stream/file path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    return parser.parse_args()


def get_source(source_arg: str):
    if source_arg.isdigit():
        return int(source_arg)
    return source_arg


def get_class_color(class_id: int, color_cache: Dict[int, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    if class_id not in color_cache:
        color_cache[class_id] = (
            (37 * class_id) % 255,
            (17 * class_id) % 255,
            (29 * class_id) % 255,
        )
    return color_cache[class_id]


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)

    source = get_source(args.source)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {args.source}")

    class_colors: Dict[int, Tuple[int, int, int]] = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(
            source=frame,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            verbose=False,
        )

        result = results[0]
        names = result.names

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()

                color = get_class_color(class_id, class_colors)
                label = f"{names[class_id]} {confidence:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                top = max(y1, label_size[1] + 10)
                cv2.rectangle(
                    frame,
                    (x1, top - label_size[1] - 10),
                    (x1 + label_size[0] + 6, top + baseline - 8),
                    color,
                    -1,
                )
                cv2.putText(
                    frame,
                    label,
                    (x1 + 3, top - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        cv2.imshow("YOLOv8 Live Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
