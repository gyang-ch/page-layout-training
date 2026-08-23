#!/usr/bin/env python3
"""Evaluate one YOLO checkpoint on the fixed test split and save metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))
    metrics = model.val(
        data=str(args.data),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=args.workers,
        conf=0.001,
        iou=0.7,
        max_det=300,
        augment=False,
        save_json=True,
        save_txt=True,
        save_conf=True,
        plots=True,
        project=str(args.output_dir.parent),
        name=args.output_dir.name,
        exist_ok=True,
        verbose=True,
    )

    summary = {
        "split": "test",
        "checkpoint": str(args.model.resolve()),
        "data_config": str(args.data.resolve()),
        "metrics": {key: float(value) for key, value in metrics.results_dict.items()},
        "per_class_mAP50_95": {
            str(metrics.names[index]): float(value)
            for index, value in enumerate(metrics.box.maps)
        },
        "speed_ms_per_image": {
            key: float(value) for key, value in metrics.speed.items()
        },
    }
    (args.output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
