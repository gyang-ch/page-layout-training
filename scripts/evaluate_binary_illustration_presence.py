#!/usr/bin/env python3
"""Evaluate detector outputs as page-level illustration-presence decisions.

Ground truth is positive when a COCO image has at least one annotation in the
``illustration`` category. A model decision is positive when at least one
predicted illustration has a score greater than or equal to the confidence
threshold. Text-block predictions and bounding-box overlap are intentionally
ignored because the evaluated question is only: "Does this page contain an
illustration?"

The script consumes predictions already produced by
``run_test_evaluation_queue.sh``. It does not load checkpoints or run inference.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


DEFAULT_SIZES = (200, 400, 600, 800, 1000, 1152)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=(0.10, 0.25, 0.50),
        help="Report a sensitivity analysis at these fixed thresholds.",
    )
    parser.add_argument(
        "--primary-threshold",
        type=float,
        default=0.25,
        help="Threshold used for the primary summary and per-image audit.",
    )
    return parser.parse_args()


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def load_ground_truth(path: Path) -> tuple[dict[int, str], set[int], int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    images = {int(image["id"]): image["file_name"] for image in data["images"]}
    if len(images) != 263:
        raise ValueError(f"Expected 263 fixed test images; found {len(images)}")
    if len(set(images.values())) != len(images):
        raise ValueError("Duplicate filenames found in the fixed test annotations")

    illustration_ids = [
        int(category["id"])
        for category in data["categories"]
        if category["name"].strip().lower() == "illustration"
    ]
    if len(illustration_ids) != 1:
        raise ValueError(f"Expected one illustration category; found {illustration_ids}")
    illustration_id = illustration_ids[0]
    positive_ids = {
        int(annotation["image_id"])
        for annotation in data["annotations"]
        if int(annotation["category_id"]) == illustration_id
    }
    unknown = positive_ids - set(images)
    if unknown:
        raise ValueError(f"Annotations reference unknown image IDs: {sorted(unknown)[:5]}")
    return images, positive_ids, illustration_id


def load_max_scores(
    path: Path,
    family: str,
    images: dict[int, str],
    illustration_id: int,
) -> dict[int, float]:
    predictions = json.loads(path.read_text(encoding="utf-8"))
    filename_to_id = {filename: image_id for image_id, filename in images.items()}
    scores: dict[int, float] = defaultdict(float)
    unknown: set[str] = set()

    for prediction in predictions:
        if int(prediction["category_id"]) != illustration_id:
            continue
        if family == "yolo":
            filename = prediction.get("file_name")
            image_id = filename_to_id.get(filename)
            if image_id is None:
                unknown.add(str(filename))
                continue
        else:
            image_id = int(prediction["image_id"])
            if image_id not in images:
                unknown.add(str(image_id))
                continue
        score = float(prediction["score"])
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Prediction score outside [0, 1] in {path}: {score}")
        scores[image_id] = max(scores[image_id], score)

    if unknown:
        raise ValueError(f"Unmapped prediction images in {path}: {sorted(unknown)[:5]}")
    return {image_id: scores.get(image_id, 0.0) for image_id in images}


def metrics_for_threshold(
    positive_ids: set[int], max_scores: dict[int, float], threshold: float
) -> dict[str, int | float | None]:
    predicted_positive = {
        image_id for image_id, score in max_scores.items() if score >= threshold
    }
    all_ids = set(max_scores)
    negative_ids = all_ids - positive_ids
    tp = len(positive_ids & predicted_positive)
    fn = len(positive_ids - predicted_positive)
    fp = len(negative_ids & predicted_positive)
    tn = len(negative_ids - predicted_positive)
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "threshold": threshold,
        "test_images": len(all_ids),
        "positive_pages": len(positive_ids),
        "negative_pages": len(negative_ids),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "false_negative_rate": ratio(fn, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "accuracy": ratio(tp + tn, len(all_ids)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.primary_threshold <= 1.0:
        raise ValueError("Primary threshold must be in [0, 1]")
    thresholds = sorted(set(args.thresholds) | {args.primary_threshold})
    if any(not 0.0 <= threshold <= 1.0 for threshold in thresholds):
        raise ValueError("All thresholds must be in [0, 1]")

    images, positive_ids, illustration_id = load_ground_truth(args.annotations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    primary_rows: list[dict] = []
    per_image_rows: list[dict] = []

    for family, model_name in (("yolo", "YOLO11m"), ("dino", "DINO-R50")):
        for size in args.sizes:
            run_dir = args.test_root / family / f"train_{size}_seed_{args.seed}"
            prediction_path = run_dir / (
                "predictions.json" if family == "yolo" else "coco_predictions.bbox.json"
            )
            if not prediction_path.is_file():
                raise FileNotFoundError(f"Missing predictions: {prediction_path}")
            max_scores = load_max_scores(
                prediction_path, family, images, illustration_id
            )
            for threshold in thresholds:
                row = {
                    "model": model_name,
                    "family": family,
                    "training_images": size,
                    "seed": args.seed,
                    **metrics_for_threshold(positive_ids, max_scores, threshold),
                }
                all_rows.append(row)
                if threshold == args.primary_threshold:
                    primary_rows.append(row)

            for image_id, filename in sorted(images.items()):
                actual = image_id in positive_ids
                score = max_scores[image_id]
                predicted = score >= args.primary_threshold
                outcome = "TP" if actual and predicted else "FN" if actual else "FP" if predicted else "TN"
                per_image_rows.append(
                    {
                        "model": model_name,
                        "training_images": size,
                        "seed": args.seed,
                        "threshold": args.primary_threshold,
                        "image_id": image_id,
                        "file_name": filename,
                        "ground_truth_has_illustration": int(actual),
                        "max_illustration_score": score,
                        "predicted_has_illustration": int(predicted),
                        "outcome": outcome,
                    }
                )

    write_csv(args.output_dir / "binary_illustration_presence_all_thresholds.csv", all_rows)
    write_csv(args.output_dir / "binary_illustration_presence_primary.csv", primary_rows)
    write_csv(args.output_dir / "binary_illustration_presence_per_image_primary.csv", per_image_rows)
    metadata = {
        "question": "Does this page contain at least one illustration?",
        "ground_truth_rule": "Positive if the COCO image has >=1 illustration annotation.",
        "prediction_rule": "Positive if max illustration confidence >= threshold.",
        "box_IoU_used": False,
        "text_block_predictions_used": False,
        "primary_threshold": args.primary_threshold,
        "reported_thresholds": thresholds,
        "test_images": len(images),
        "positive_pages": len(positive_ids),
        "negative_pages": len(images) - len(positive_ids),
        "seed": args.seed,
        "annotations": str(args.annotations.resolve()),
        "test_root": str(args.test_root.resolve()),
    }
    (args.output_dir / "binary_illustration_presence_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Fixed test set: {len(images)} pages")
    print(f"Illustration present: {len(positive_ids)}; absent: {len(images) - len(positive_ids)}")
    print(f"Primary confidence threshold: {args.primary_threshold:.2f}\n")
    print("model       train   TP  FP  TN  FN  precision  recall    F1     FNR")
    for row in primary_rows:
        print(
            f"{row['model']:<11} {row['training_images']:>5} "
            f"{row['TP']:>4} {row['FP']:>3} {row['TN']:>3} {row['FN']:>3} "
            f"{row['precision']:.4f}    {row['recall']:.4f}  "
            f"{row['F1']:.4f}  {row['false_negative_rate']:.4f}"
        )
    print(f"\nWrote results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
