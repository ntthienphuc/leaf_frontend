#!/usr/bin/env python3
"""Create a deduplicated train/val/test folder from the Kaggle pepper dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
from pathlib import Path


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    classes = sorted(path for path in args.source.iterdir() if path.is_dir())
    if not classes:
        raise SystemExit(f"No class folders found in {args.source}")
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for class_dir in classes:
        unique: dict[str, Path] = {}
        for image in sorted(class_dir.iterdir()):
            if image.is_file():
                unique.setdefault(digest(image), image)
        examples = list(unique.items())
        random.Random(args.seed).shuffle(examples)
        n = len(examples)
        cut1, cut2 = int(n * 0.8), int(n * 0.9)
        for split, group in (("train", examples[:cut1]), ("val", examples[cut1:cut2]), ("test", examples[cut2:])):
            target = args.output / split / class_dir.name
            target.mkdir(parents=True, exist_ok=True)
            for index, (sha256, source) in enumerate(group):
                destination = target / f"{index:05d}_{source.name}"
                shutil.copy2(source, destination)
                rows.append({"split": split, "class": class_dir.name, "source": str(source), "sha256": sha256})

    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["split", "class", "source", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Prepared {len(rows)} unique images across {len(classes)} classes at {args.output}")


if __name__ == "__main__":
    main()
