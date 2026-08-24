#!/usr/bin/env python3
"""Shared image discovery and drawing helpers for external-corpus inference."""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
CLASS_NAMES = {0: "illustration", 1: "text_block"}
CLASS_COLORS = {0: (0, 114, 178), 1: (213, 94, 0)}


def discover_images(root: Path) -> list[Path]:
    """Return a stable recursive list while retaining book subdirectories.

    ``os.walk(..., followlinks=True)`` is intentional: external-corpus queues
    can stage selected book directories as symlinks without copying images.
    """
    images = []
    for directory, _, filenames in os.walk(root, followlinks=True):
        directory_path = Path(directory)
        images.extend(
            directory_path / filename
            for filename in filenames
            if Path(filename).suffix.lower() in IMAGE_SUFFIXES
        )
    images.sort()
    if not images:
        raise ValueError(f"No supported images found under {root}")
    return images


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_predictions(
    source: Path,
    destination: Path,
    detections: list[dict],
) -> None:
    """Draw class, confidence, and bounding boxes on one source image."""
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    short_side = min(image.size)
    line_width = max(3, round(short_side / 300))
    font = load_font(max(15, round(short_side / 45)))
    padding = max(2, line_width)

    for detection in sorted(detections, key=lambda item: item["score"]):
        class_id = int(detection["class_id"])
        color = CLASS_COLORS[class_id]
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        x1 = max(0.0, min(float(x1), image.width - 1))
        y1 = max(0.0, min(float(y1), image.height - 1))
        x2 = max(x1, min(float(x2), image.width - 1))
        y2 = max(y1, min(float(y2), image.height - 1))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)

        label = f"{CLASS_NAMES[class_id]} {float(detection['score']):.2f}"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        label_width = right - left + 2 * padding
        label_height = bottom - top + 2 * padding
        label_x = min(x1, max(0, image.width - label_width))
        label_y = y1 - label_height if y1 >= label_height else y1
        draw.rectangle(
            (label_x, label_y, label_x + label_width, label_y + label_height),
            fill=color,
        )
        draw.text(
            (label_x + padding, label_y + padding - top),
            label,
            fill="white",
            font=font,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, quality=92, optimize=True)
