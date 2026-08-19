#!/usr/bin/env python3
"""Train a one-class leaf detector after preparing LeavesBank."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/prepared/leavesbank_detect/data.yaml"))
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--device", default=None, help="For example 0 for Colab GPU")
    parser.add_argument("--project", default="runs/leaf_detect")
    args = parser.parse_args()

    if not args.data.is_file():
        raise SystemExit(f"Dataset config does not exist: {args.data}")
    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name="leavesbank_yolo26n_detect",
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
    )


if __name__ == "__main__":
    main()
