#!/usr/bin/env python3
"""Train YOLO26 / YOLO11 leaf segmentation model on LeavesBank dataset."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO leaf segmentation model")
    parser.add_argument("--data", type=Path, default=Path("/root/leavesbank_seg/data.yaml"))
    parser.add_argument("--model", default="yolo26n-seg.pt", help="Base model checkpoint e.g. yolo26n-seg.pt, yolo11n-seg.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0", help="CUDA device index, e.g. 0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="/workspace/runs/leaf_seg")
    parser.add_argument("--name", default="leavesbank_yolo26n_seg")
    args = parser.parse_args()

    if not args.data.is_file():
        raise SystemExit(f"Dataset config does not exist: {args.data}")

    print(f"=== Starting YOLO Segmentation Training ===")
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}, Imgsz: {args.imgsz}, Batch: {args.batch}, Device: {args.device}")

    model = YOLO(args.model)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        pretrained=True,
        patience=25,
        cos_lr=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=0.5,
        verbose=True,
        save=True,
    )

    print("\n=== Training Completed Successfully! ===")
    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    if best_weights.is_file():
        dest = Path("/workspace/models/leaf_detector_yolo26n_seg.pt")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, dest)
        print(f"Backup best model to network volume: {dest}")


if __name__ == "__main__":
    main()
