#!/usr/bin/env python3
"""Convert LeavesBank LabelMe/AnyLabeling JSON annotations into YOLO Segmentation format."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from PIL import Image
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_image_for_json(json_path: Path, image_name_hint: str | None) -> Path | None:
    candidates = []
    if image_name_hint:
        candidates.append(json_path.parent / image_name_hint)
        candidates.append(json_path.parent.parent / "images" / image_name_hint)
        candidates.append(json_path.parent.parent / image_name_hint)
    candidates.extend(json_path.with_suffix(ext) for ext in IMAGE_EXTENSIONS)
    stem = json_path.stem
    for ext in IMAGE_EXTENSIONS:
        candidates.append(json_path.parent.parent / "images" / f"{stem}{ext}")
    return next((p for p in candidates if p.is_file()), None)


def parse_polygon(shape: dict, width: int, height: int) -> list[float] | None:
    label = str(shape.get("label", "")).lower()
    label_clean = "".join(c for c in label if c.isalnum() or c == "_")
    if not label_clean.startswith("leaf"):
        return None

    raw_points = shape.get("points") or []
    if len(raw_points) < 3:
        return None

    parsed_coords = []
    for pt in raw_points:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            x, y = float(pt[0]), float(pt[1])
        elif isinstance(pt, str):
            parts = pt.replace(",", " ").split()
            if len(parts) < 2:
                continue
            x, y = float(parts[0]), float(parts[1])
        else:
            continue
        norm_x = min(1.0, max(0.0, x / width))
        norm_y = min(1.0, max(0.0, y / height))
        parsed_coords.extend([norm_x, norm_y])

    if len(parsed_coords) < 6:
        return None

    return parsed_coords


def process_sample(args_tuple):
    json_path, split, output_dir = args_tuple
    try:
        with open(json_path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)

        img_path = find_image_for_json(json_path, data.get("imagePath"))
        if not img_path:
            return 0

        w = data.get("imageWidth")
        h = data.get("imageHeight")
        if not w or not h or w <= 0 or h <= 0:
            with Image.open(img_path) as img:
                w, h = img.size

        polygons = []
        for shape in data.get("shapes", []):
            poly = parse_polygon(shape, w, h)
            if poly:
                poly_str = "0 " + " ".join(f"{coord:.6f}" for coord in poly)
                polygons.append(poly_str)

        if not polygons:
            return 0

        unique_name = f"{json_path.parent.parent.name}_{json_path.parent.name}_{img_path.name}".replace(" ", "_")
        dest_img = output_dir / "images" / split / unique_name
        dest_lbl = output_dir / "labels" / split / f"{dest_img.stem}.txt"

        dest_img.parent.mkdir(parents=True, exist_ok=True)
        dest_lbl.parent.mkdir(parents=True, exist_ok=True)

        try:
            os.link(img_path, dest_img)
        except Exception:
            shutil.copy2(img_path, dest_img)

        with open(dest_lbl, "w", encoding="utf-8") as f:
            f.write("\n".join(polygons) + "\n")

        return len(polygons)
    except Exception:
        return 0


def main():
    parser = argparse.ArgumentParser(description="Convert LeavesBank to YOLO Seg format")
    parser.add_argument("--source", type=Path, required=True, help="Path to LeavesBank dataset root")
    parser.add_argument("--output", type=Path, default=Path("/root/leavesbank_seg"), help="Output YOLO dataset path")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--workers", type=int, default=16, help="Number of workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print(f"Scanning JSON annotations in: {args.source}")
    json_files = sorted(list(args.source.glob("**/*.json")))
    print(f"Found {len(json_files)} total annotation files.")

    if not json_files:
        raise SystemExit("No JSON files found in source directory!")

    random.seed(args.seed)
    random.shuffle(json_files)

    num_val = int(len(json_files) * args.val_ratio)
    val_set = set(json_files[:num_val])

    tasks = []
    for p in json_files:
        split = "val" if p in val_set else "train"
        tasks.append((p, split, args.output))

    print(f"Processing {len(tasks)} samples with {args.workers} workers...")
    total_instances = 0
    total_images = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for res in tqdm(executor.map(process_sample, tasks), total=len(tasks)):
            if res > 0:
                total_instances += res
                total_images += 1

    print(f"Done! Extracted {total_images} images with {total_instances} leaf segmentation instances.")

    # Create data.yaml
    data_yaml = args.output / "data.yaml"
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write(f"""path: {args.output.resolve()}
train: images/train
val: images/val

names:
  0: leaf
""")
    print(f"Created data config at: {data_yaml}")


if __name__ == "__main__":
    main()
