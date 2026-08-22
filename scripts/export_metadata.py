#!/usr/bin/env python3
"""Build the Git-tracked package metadata from the local Roboflow exports.

This script copies only small text/JSON metadata. Images and YOLO labels remain
in their original directories and are represented by a SHA-256 inventory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


SPLITS = ("train", "valid", "test")
SIZES = (200, 400, 600, 800, 1000, 1152)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    default_workspace = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=default_workspace)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    repo = Path(__file__).resolve().parents[1]
    yolo = workspace / "Page_layout.v2i.yolov11"
    coco = workspace / "Page_layout.v2i.coco-mmdetection"
    experiments = workspace / "Page_layout_experiments"

    for required in (yolo, coco, experiments):
        if not required.is_dir():
            raise FileNotFoundError(f"Required source directory not found: {required}")

    manifests_out = repo / "manifests"
    annotations_out = repo / "annotations" / "coco"
    metadata_out = repo / "metadata"
    manifests_out.mkdir(parents=True, exist_ok=True)
    annotations_out.mkdir(parents=True, exist_ok=True)
    metadata_out.mkdir(parents=True, exist_ok=True)

    for size in SIZES:
        source = experiments / "manifests" / f"train_{size}.txt"
        names = source.read_text(encoding="utf-8").splitlines()
        if len(names) != size or len(set(names)) != size:
            raise ValueError(f"Invalid source manifest: {source}")
        shutil.copy2(source, manifests_out / source.name)
        shutil.copy2(
            experiments / "coco_mmdetection" / f"train_{size}.json",
            annotations_out / f"train_{size}.json",
        )

    # Preserve the full original COCO annotations for provenance and fixed
    # validation/test evaluation. These files are small enough for ordinary Git.
    for split in SPLITS:
        source = coco / split / "_annotations.coco.json"
        destination = annotations_out / ("train_full.json" if split == "train" else f"{split}.json")
        shutil.copy2(source, destination)

    files: dict[str, dict[str, int | str]] = {}
    split_summary: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        image_dir = yolo / split / "images"
        label_dir = yolo / split / "labels"
        images = sorted(
            path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        labels = sorted(label_dir.glob("*.txt"))
        if {path.stem for path in images} != {path.stem for path in labels}:
            raise ValueError(f"YOLO image/label mismatch in split: {split}")

        for kind, paths in (("images", images), ("labels", labels)):
            for path in paths:
                normalized = f"{kind}/{split}/{path.name}"
                files[normalized] = {"size": path.stat().st_size, "sha256": sha256(path)}
        split_summary[split] = {"images": len(images), "labels": len(labels)}

    validation_report = json.loads(
        (experiments / "validation_report.json").read_text(encoding="utf-8")
    )
    inventory = {
        "dataset": "page-layout",
        "version": "v2-20260822",
        "selection_seed": validation_report["seed"],
        "classes": ["illustration", "text_block"],
        "normalized_layout": "images/<split> and labels/<split>",
        "splits": split_summary,
        "files": files,
    }
    (metadata_out / "dataset_inventory.json").write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(experiments / "validation_report.json", metadata_out / "subset_validation_report.json")

    print(f"Exported {len(SIZES)} manifests and subset annotation files")
    print(f"Inventoried {len(files)} unique image/label files")
    print(f"Package root: {repo}")


if __name__ == "__main__":
    main()

