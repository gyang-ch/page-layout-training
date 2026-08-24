#!/usr/bin/env python3
"""Run one MMDetection DINO checkpoint on a recursive external image corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmdet.apis import inference_detector, init_detector

from inference_visualization import CLASS_NAMES, discover_images, draw_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = discover_images(args.input_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotated_root = args.output_dir / "annotated"
    predictions_path = args.output_dir / "predictions.jsonl"
    model = init_detector(
        str(args.config), str(args.checkpoint), device=args.device
    )
    class_counts = {name: 0 for name in CLASS_NAMES.values()}
    pages_with_detections = 0

    with predictions_path.open("w", encoding="utf-8") as output:
        for index, image_path in enumerate(images, start=1):
            result = inference_detector(model, str(image_path))
            instances = result.pred_instances.cpu()
            detections = []
            for bbox, score, label in zip(
                instances.bboxes.tolist(),
                instances.scores.tolist(),
                instances.labels.tolist(),
            ):
                if float(score) < args.confidence:
                    continue
                class_id = int(label)
                if class_id not in CLASS_NAMES:
                    raise ValueError(f"Unexpected DINO class ID: {class_id}")
                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": CLASS_NAMES[class_id],
                        "score": float(score),
                        "bbox_xyxy": [float(value) for value in bbox],
                    }
                )
                class_counts[CLASS_NAMES[class_id]] += 1
            relative = image_path.relative_to(args.input_root)
            pages_with_detections += bool(detections)
            draw_predictions(image_path, annotated_root / relative, detections)
            with ImageDimensions(image_path) as dimensions:
                width, height = dimensions.size
            output.write(json.dumps({
                "source_path": relative.as_posix(),
                "width": width,
                "height": height,
                "detections": detections,
            }) + "\n")
            if index % 25 == 0 or index == len(images):
                print(f"DINO processed {index}/{len(images)}")

    summary = {
        "architecture": "DINO-R50",
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "input_root": str(args.input_root.resolve()),
        "images": len(images),
        "pages_with_detections": pages_with_detections,
        "confidence_threshold": args.confidence,
        "class_detection_counts": class_counts,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"DINO inference complete: {args.output_dir}")


class ImageDimensions:
    """Open only long enough to read dimensions, then close deterministically."""

    def __init__(self, path: Path):
        self.path = path
        self.image = None

    def __enter__(self):
        from PIL import Image

        self.image = Image.open(self.path)
        return self.image

    def __exit__(self, exc_type, exc_value, traceback):
        self.image.close()


if __name__ == "__main__":
    main()
