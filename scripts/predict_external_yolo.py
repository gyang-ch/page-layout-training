#!/usr/bin/env python3
"""Run one YOLO page-layout checkpoint on a recursive external image corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from inference_visualization import CLASS_NAMES, discover_images, draw_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = discover_images(args.input_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotated_root = args.output_dir / "annotated"
    predictions_path = args.output_dir / "predictions.jsonl"
    model = YOLO(str(args.model))
    class_counts = {name: 0 for name in CLASS_NAMES.values()}
    pages_with_detections = 0

    with predictions_path.open("w", encoding="utf-8") as output:
        for start in range(0, len(images), args.batch_size):
            batch = images[start:start + args.batch_size]
            results = model.predict(
                source=[str(path) for path in batch],
                imgsz=args.imgsz,
                conf=args.confidence,
                iou=args.iou,
                max_det=300,
                device=args.device,
                verbose=False,
            )
            if len(results) != len(batch):
                raise RuntimeError("YOLO returned a different number of results than inputs")
            for image_path, result in zip(batch, results):
                relative = image_path.relative_to(args.input_root)
                detections = []
                if result.boxes is not None:
                    bboxes = result.boxes.xyxy.detach().cpu().tolist()
                    scores = result.boxes.conf.detach().cpu().tolist()
                    labels = result.boxes.cls.detach().cpu().tolist()
                    for bbox, score, label in zip(bboxes, scores, labels):
                        class_id = int(label)
                        if class_id not in CLASS_NAMES:
                            raise ValueError(f"Unexpected YOLO class ID: {class_id}")
                        detections.append(
                            {
                                "class_id": class_id,
                                "class_name": CLASS_NAMES[class_id],
                                "score": float(score),
                                "bbox_xyxy": [float(value) for value in bbox],
                            }
                        )
                        class_counts[CLASS_NAMES[class_id]] += 1
                pages_with_detections += bool(detections)
                draw_predictions(image_path, annotated_root / relative, detections)
                output.write(json.dumps({
                    "source_path": relative.as_posix(),
                    "width": int(result.orig_shape[1]),
                    "height": int(result.orig_shape[0]),
                    "detections": detections,
                }) + "\n")
            print(f"YOLO processed {min(start + len(batch), len(images))}/{len(images)}")

    summary = {
        "architecture": "YOLO11m",
        "checkpoint": str(args.model.resolve()),
        "input_root": str(args.input_root.resolve()),
        "images": len(images),
        "pages_with_detections": pages_with_detections,
        "confidence_threshold": args.confidence,
        "nms_iou_threshold": args.iou,
        "image_size": args.imgsz,
        "class_detection_counts": class_counts,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"YOLO inference complete: {args.output_dir}")


if __name__ == "__main__":
    main()

