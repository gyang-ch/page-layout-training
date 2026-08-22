#!/usr/bin/env python3
"""Verify a normalized RunPod dataset against the Git-tracked inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--layout",
        choices=("normalized", "roboflow-yolo"),
        default="normalized",
        help="Normalized uses images/<split>; Roboflow uses <split>/images",
    )
    parser.add_argument("--skip-hashes", action="store_true", help="Check paths and sizes only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    inventory = json.loads((repo / "metadata" / "dataset_inventory.json").read_text())
    dataset_root = args.dataset_root.resolve()
    missing: list[str] = []
    wrong_size: list[str] = []
    wrong_hash: list[str] = []

    for relative, expected in inventory["files"].items():
        if args.layout == "normalized":
            path = dataset_root / relative
        else:
            kind, split, filename = relative.split("/", 2)
            path = dataset_root / split / kind / filename
        if not path.is_file():
            missing.append(relative)
            continue
        if path.stat().st_size != expected["size"]:
            wrong_size.append(relative)
            continue
        if not args.skip_hashes and sha256(path) != expected["sha256"]:
            wrong_hash.append(relative)

    if missing or wrong_size or wrong_hash:
        details = {
            "missing": missing,
            "wrong_size": wrong_size,
            "wrong_sha256": wrong_hash,
        }
        raise SystemExit("Dataset verification failed:\n" + json.dumps(details, indent=2))

    mode = "paths and sizes" if args.skip_hashes else "paths, sizes, and SHA-256"
    print(
        f"Dataset verification passed ({mode}, {args.layout} layout): "
        f"{len(inventory['files'])} files"
    )
    for split, counts in inventory["splits"].items():
        print(f"  {split}: {counts['images']} images, {counts['labels']} labels")


if __name__ == "__main__":
    main()
