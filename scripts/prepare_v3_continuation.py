#!/usr/bin/env python3
"""Validate the normalized v3 dataset and generate continuation configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED = {"train": 1511, "valid": 248, "test": 263}
CLASS_NAMES = {0: "illustration", 1: "text_block"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dino-base-config", type=Path, required=True)
    parser.add_argument("--dino-init-checkpoint", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--dino-batch-size", type=int, default=2)
    parser.add_argument("--dino-num-workers", type=int, default=2)
    parser.add_argument("--dino-max-epochs", type=int, default=12)
    parser.add_argument("--dino-lr", type=float, default=2e-5)
    parser.add_argument("--dino-lr-milestone", type=int, default=10)
    return parser.parse_args()


def validate_yolo_label(path: Path) -> int:
    count = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        try:
            class_id = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError as error:
            raise ValueError(f"Invalid YOLO label {path}:{line_number}") from error
        if class_id not in CLASS_NAMES:
            raise ValueError(f"Unexpected class ID in {path}:{line_number}: {class_id}")
        # Roboflow may retain polygon coordinates for smart-polygon annotations.
        # Ultralytics detection loading converts these segments to bounding boxes.
        is_box = len(coordinates) == 4
        is_polygon = len(coordinates) >= 6 and len(coordinates) % 2 == 0
        if not (is_box or is_polygon):
            raise ValueError(f"Invalid coordinate count in {path}:{line_number}")
        if any(not 0.0 <= value <= 1.0 for value in coordinates):
            raise ValueError(f"Out-of-range coordinate in {path}:{line_number}")
        count += 1
    return count


def validate_split(root: Path, split: str) -> dict:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    coco_path = root / "annotations" / "coco" / f"{split}.json"
    images = sorted(image_dir.glob("*.jpg"))
    labels = sorted(label_dir.glob("*.txt"))
    if len(images) != EXPECTED[split] or len(labels) != EXPECTED[split]:
        raise ValueError(
            f"{split}: expected {EXPECTED[split]} images/labels; "
            f"found {len(images)} images and {len(labels)} labels"
        )
    image_names = {path.name for path in images}
    expected_label_names = {Path(name).with_suffix(".txt").name for name in image_names}
    if {path.name for path in labels} != expected_label_names:
        raise ValueError(f"{split}: YOLO image/label filename mismatch")

    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    categories = {int(item["id"]): item["name"] for item in coco["categories"]}
    if categories != {1: "illustration", 2: "text_block"}:
        raise ValueError(f"{split}: unexpected COCO categories: {categories}")
    coco_names = [item["file_name"] for item in coco["images"]]
    if len(coco_names) != EXPECTED[split] or set(coco_names) != image_names:
        raise ValueError(f"{split}: COCO image records do not match the v3 image files")
    image_ids = {int(item["id"]) for item in coco["images"]}
    if len(image_ids) != len(coco["images"]):
        raise ValueError(f"{split}: duplicate COCO image IDs")
    annotation_ids = {int(item["id"]) for item in coco["annotations"]}
    if len(annotation_ids) != len(coco["annotations"]):
        raise ValueError(f"{split}: duplicate COCO annotation IDs")
    if any(int(item["image_id"]) not in image_ids for item in coco["annotations"]):
        raise ValueError(f"{split}: COCO annotation references an unknown image")
    if any(float(item["bbox"][2]) <= 0 or float(item["bbox"][3]) <= 0 for item in coco["annotations"]):
        raise ValueError(f"{split}: non-positive COCO bounding box")

    yolo_objects = sum(validate_yolo_label(path) for path in labels)
    if yolo_objects != len(coco["annotations"]):
        raise ValueError(
            f"{split}: YOLO/COCO object-count mismatch: "
            f"{yolo_objects} versus {len(coco['annotations'])}"
        )
    return {
        "images": len(images),
        "annotations": len(coco["annotations"]),
        "empty_yolo_labels": sum(not path.read_text(encoding="utf-8").strip() for path in labels),
    }


def quoted(path: Path) -> str:
    return repr(str(path.absolute()))


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    dataset_root = args.dataset_root.resolve()
    base_config = args.dino_base_config.resolve()
    init_checkpoint = args.dino_init_checkpoint.resolve()
    if not base_config.is_file():
        raise FileNotFoundError(base_config)
    if not init_checkpoint.is_file():
        raise FileNotFoundError(init_checkpoint)
    if not 0 < args.dino_lr_milestone < args.dino_max_epochs:
        raise ValueError("DINO LR milestone must fall inside the training schedule")

    validation = {
        split: validate_split(dataset_root, split)
        for split in ("train", "valid", "test")
    }
    runtime = args.runtime_root.resolve() if args.runtime_root else repo / ".runtime_v3"
    (runtime / "yolo").mkdir(parents=True, exist_ok=True)
    (runtime / "dino").mkdir(parents=True, exist_ok=True)

    yolo_config = runtime / "yolo" / "train_1511_from_1152.yaml"
    yolo_config.write_text(
        "\n".join((
            f"path: {dataset_root}",
            "train: images/train",
            "val: images/valid",
            "test: images/test",
            "",
            "nc: 2",
            "names:",
            "  0: illustration",
            "  1: text_block",
            "",
        )),
        encoding="utf-8",
    )

    dino_config = runtime / "dino" / "train_1511_from_1152.py"
    train_json = dataset_root / "annotations" / "coco" / "train.json"
    valid_json = dataset_root / "annotations" / "coco" / "valid.json"
    test_json = dataset_root / "annotations" / "coco" / "test.json"
    dino_config.write_text(
        f"""# Generated by prepare_v3_continuation.py; do not commit this file.
_base_ = {quoted(base_config)}

dataset_type = 'CocoDataset'
data_root = {repr(str(dataset_root) + '/')}
metainfo = dict(classes=('illustration', 'text_block'))
model = dict(bbox_head=dict(num_classes=2))

# Warm-start model weights from the old 1,152-image best checkpoint, but begin
# a new optimizer and schedule because both the dataset and annotations changed.
load_from = {quoted(init_checkpoint)}
resume = False
randomness = dict(seed={args.seed}, deterministic=False)

max_epochs = {args.dino_max_epochs}
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
param_scheduler = [dict(
    type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True,
    milestones=[{args.dino_lr_milestone}], gamma=0.1)]
optim_wrapper = dict(optimizer=dict(lr={args.dino_lr}))

train_dataloader = dict(
    batch_size={args.dino_batch_size}, num_workers={args.dino_num_workers},
    dataset=dict(
        type=dataset_type, data_root=data_root, ann_file={quoted(train_json)},
        data_prefix=dict(img='images/train/'), metainfo=metainfo))
val_dataloader = dict(
    batch_size=1, num_workers={args.dino_num_workers},
    dataset=dict(
        type=dataset_type, data_root=data_root, ann_file={quoted(valid_json)},
        data_prefix=dict(img='images/valid/'), metainfo=metainfo))
test_dataloader = dict(
    batch_size=1, num_workers={args.dino_num_workers},
    dataset=dict(
        type=dataset_type, data_root=data_root, ann_file={quoted(test_json)},
        data_prefix=dict(img='images/test/'), metainfo=metainfo))
val_evaluator = dict(ann_file={quoted(valid_json)})
test_evaluator = dict(ann_file={quoted(test_json)})
default_hooks = dict(checkpoint=dict(
    type='CheckpointHook', interval=1, save_best='coco/bbox_mAP',
    rule='greater', max_keep_ckpts=3))
""",
        encoding="utf-8",
    )

    summary = {
        "dataset": "Page_layout v3",
        "dataset_root": str(dataset_root),
        "seed": args.seed,
        "initialization": {
            "YOLO": "/workspace/outputs/yolo/train_1152_seed_20260822/weights/best.pt",
            "DINO": str(init_checkpoint),
        },
        "validation": validation,
        "yolo_config": str(yolo_config),
        "dino_config": str(dino_config),
        "dino_schedule": {
            "epochs": args.dino_max_epochs,
            "lr": args.dino_lr,
            "milestone": args.dino_lr_milestone,
        },
    }
    (runtime / "v3_runtime_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2))
    print(f"V3 runtime generated in {runtime}")


if __name__ == "__main__":
    main()
