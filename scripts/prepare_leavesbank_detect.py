#!/usr/bin/env python3
"""Convert LeavesBank AnyLabeling polygons into a one-class YOLO detect dataset."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_image(annotation: Path, image_path: str | None) -> Path | None:
    candidates = []
    if image_path:
        candidates.append(annotation.parent / image_path)
        # LeavesBank stores images and AnyLabeling JSON in sibling directories.
        candidates.append(annotation.parent.parent / "images" / image_path)
        candidates.append(annotation.parent.parent / image_path)
    candidates.extend(annotation.with_suffix(ext) for ext in IMAGE_SUFFIXES)
    return next((path for path in candidates if path.is_file()), None)


def to_box(shape: dict, width: int, height: int) -> tuple[float, float, float, float] | None:
    normalized_label = "".join(char for char in str(shape.get("label", "")).lower() if char.isalpha())
    # The released data spells the secondary class as both Leaf_Secondarly and leaf_secondary.
    if not normalized_label.startswith("leaf"):
        return None
    points = shape.get("points") or []
    if len(points) < 2:
        return None
    parsed_points = []
    for point in points:
        if isinstance(point, str):
            values = point.replace(",", " ").split()
            if len(values) < 2:
                continue
            parsed_points.append((float(values[0]), float(values[1])))
        else:
            parsed_points.append((float(point[0]), float(point[1])))
    if len(parsed_points) < 2:
        return None
    xs = [point[0] for point in parsed_points]
    ys = [point[1] for point in parsed_points]
    x1, x2 = max(0.0, min(xs)), min(float(width), max(xs))
    y1, y2 = max(0.0, min(ys)), min(float(height), max(ys))
    if x2 <= x1 or y2 <= y1:
        return None
    return ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height), (x2 - x1) / width, (y2 - y1) / height)


def load_official_splits(source: Path) -> dict[str, str] | None:
    """Map image basenames to the dataset's published train/val/test split."""
    split_map: dict[str, str] = {}
    for split in ("train", "val", "test"):
        split_file = source / f"{split}_images.txt"
        if not split_file.is_file():
            return None
        for name in split_file.read_text(encoding="utf-8").splitlines():
            normalized = Path(name.strip()).name.casefold()
            if normalized:
                split_map[normalized] = split
    return split_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Extracted LeavesBank root")
    parser.add_argument("--output", type=Path, required=True, help="YOLO detection output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="Optional cap for a quick experiment")
    args = parser.parse_args()

    annotations = sorted(args.source.rglob("*.json"))
    official_splits = load_official_splits(args.source)
    records: list[tuple[Path, list[tuple[float, float, float, float]], str | None]] = []
    skipped = Counter()
    for annotation in annotations:
        try:
            payload = json.loads(annotation.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            skipped["invalid_json"] += 1
            continue
        image = find_image(annotation, payload.get("imagePath"))
        if image is None:
            skipped["image_missing"] += 1
            continue
        width = int(payload.get("imageWidth", 0))
        height = int(payload.get("imageHeight", 0))
        if width <= 0 or height <= 0:
            with Image.open(image) as opened:
                width, height = opened.size
        boxes = [box for shape in payload.get("shapes", []) if (box := to_box(shape, width, height))]
        if not boxes:
            skipped["no_leaf"] += 1
            continue
        split = official_splits.get(image.name.casefold()) if official_splits else None
        if official_splits and split is None:
            skipped["not_in_official_split"] += 1
            continue
        records.append((image, boxes, split))

    if not records:
        raise SystemExit("No AnyLabeling leaf annotations found. Inspect the extracted dataset layout first.")
    rng = random.Random(args.seed)
    if official_splits:
        splits = {name: [] for name in ("train", "val", "test")}
        seen_names: set[str] = set()
        for image, boxes, split in records:
            # The source includes duplicate disease folders. Keep one image per published name.
            image_name = image.name.casefold()
            if image_name in seen_names:
                skipped["duplicate_image"] += 1
                continue
            seen_names.add(image_name)
            splits[split].append((image, boxes))
        for examples in splits.values():
            rng.shuffle(examples)
        if args.limit:
            total = sum(len(examples) for examples in splits.values())
            for name, examples in splits.items():
                keep = max(1, round(args.limit * len(examples) / total))
                splits[name] = examples[:keep]
    else:
        fallback_records = [(image, boxes) for image, boxes, _ in records]
        rng.shuffle(fallback_records)
        if args.limit:
            fallback_records = fallback_records[: args.limit]
        count = len(fallback_records)
        split_end = (int(count * 0.8), int(count * 0.9))
        splits = {
            "train": fallback_records[: split_end[0]],
            "val": fallback_records[split_end[0] : split_end[1]],
            "test": fallback_records[split_end[1] :],
        }

    for split, examples in splits.items():
        image_dir = args.output / "images" / split
        label_dir = args.output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for index, (image, boxes) in enumerate(examples):
            # Sequential names avoid collisions across the source datasets.
            name = f"leavesbank_{index:06d}{image.suffix.lower()}"
            shutil.copy2(image, image_dir / name)
            rows = [f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for xc, yc, w, h in boxes]
            (label_dir / f"leavesbank_{index:06d}.txt").write_text("\n".join(rows) + "\n", encoding="ascii")

    yaml = "path: " + str(args.output.resolve()).replace("\\", "/") + "\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: leaf\n"
    (args.output / "data.yaml").write_text(yaml, encoding="ascii")
    print("Using official split" if official_splits else "Using seeded fallback split")
    print(f"Prepared {sum(len(items) for items in splits.values())} images: " + ", ".join(f"{name}={len(items)}" for name, items in splits.items()))
    print("Skipped: " + ", ".join(f"{key}={value}" for key, value in sorted(skipped.items())))


if __name__ == "__main__":
    main()
