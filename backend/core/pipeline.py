from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from ultralytics import YOLO

from .classifier import DiseaseClassifier


class LeafDiseasePipeline:
    def __init__(
        self,
        leaf_detector_path: Path,
        disease_classifier_path: Path,
        tracker_config_path: Path,
        device: str = "auto",
        detector_imgsz: int = 640,
        detector_conf: float = 0.25,
        detector_iou: float = 0.5,
        classifier_imgsz: int = 320,
        crop_padding: float = 0.08,
        smoothing_window: int = 7,
    ) -> None:
        self.device = self._resolve_device(device)
        self.detector = YOLO(str(leaf_detector_path))
        self.classifier = DiseaseClassifier(
            checkpoint_path=disease_classifier_path,
            device=self.device,
            image_size=classifier_imgsz,
        )
        self.tracker_config_path = Path(tracker_config_path)
        self.detector_imgsz = detector_imgsz
        self.detector_conf = detector_conf
        self.detector_iou = detector_iou
        self.crop_padding = crop_padding
        self.smoothing_window = smoothing_window
        self._history: dict[int | str, deque[dict[str, float]]] = defaultdict(
            lambda: deque(maxlen=smoothing_window)
        )
        self._lock = Lock()

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                return "cpu"
        return device

    def process_frame(self, frame_bgr: np.ndarray, use_tracker: bool = True) -> dict:
        height, width = frame_bgr.shape[:2]

        with self._lock:
            if use_tracker:
                results = self.detector.track(
                    source=frame_bgr,
                    persist=True,
                    tracker=str(self.tracker_config_path),
                    imgsz=self.detector_imgsz,
                    conf=self.detector_conf,
                    iou=self.detector_iou,
                    device=self.device,
                    verbose=False,
                )
            else:
                results = self.detector.predict(
                    source=frame_bgr,
                    imgsz=self.detector_imgsz,
                    conf=self.detector_conf,
                    iou=self.detector_iou,
                    device=self.device,
                    verbose=False,
                )

        result = results[0]
        detections = []
        if result.boxes is None or len(result.boxes) == 0:
            return {
                "frame": {"width": width, "height": height},
                "detections": detections,
            }

        boxes = result.boxes.xyxy.detach().cpu().numpy()
        detector_conf = result.boxes.conf.detach().cpu().numpy()
        ids = result.boxes.id.detach().cpu().numpy().astype(int) if result.boxes.id is not None else None

        # Check if segment masks exist in result
        has_masks = result.masks is not None and len(result.masks) > 0
        masks_data = result.masks.data if has_masks else None
        masks_xy = result.masks.xy if has_masks else None

        for index, box in enumerate(boxes):
            x1, y1, x2, y2 = self._pad_box(box, width, height)
            
            # Apply binary mask with neutral gray fill (128, 128, 128)
            poly_list = None
            if has_masks and masks_data is not None and index < len(masks_data):
                mask_np = masks_data[index].detach().cpu().numpy()
                if mask_np.shape[:2] != (height, width):
                    mask_resized = cv2.resize(mask_np, (width, height), interpolation=cv2.INTER_NEAREST) > 0.5
                else:
                    mask_resized = mask_np > 0.5
                
                # Canvas filled with neutral gray (128, 128, 128)
                masked_canvas = np.full_like(frame_bgr, (128, 128, 128), dtype=np.uint8)
                # Paste leaf pixels on the gray background
                masked_canvas[mask_resized] = frame_bgr[mask_resized]
                crop = masked_canvas[y1:y2, x1:x2]
                
                if masks_xy is not None and index < len(masks_xy):
                    poly_list = masks_xy[index].tolist() # list of [x, y] float coordinates
            else:
                crop = frame_bgr[y1:y2, x1:x2]

            # In case the crop is empty, skip
            if crop.size == 0:
                continue

            disease = self.classifier.predict(crop)
            track_id: int | None = int(ids[index]) if ids is not None else None
            stable = self._smooth_prediction(track_id if track_id is not None else f"det-{index}", disease)

            detections.append(
                {
                    "track_id": track_id,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "bbox_norm_xywh": self._xyxy_to_norm_xywh(x1, y1, x2, y2, width, height),
                    "mask_xy": poly_list,
                    "leaf_confidence": float(detector_conf[index]),
                    "disease": disease,
                    "stable_disease": stable,
                }
            )

        return {
            "frame": {"width": width, "height": height},
            "detections": detections,
        }

    def reset_tracker(self) -> None:
        with self._lock:
            self.detector.predictor = None
            self._history.clear()

    def _smooth_prediction(self, key: int | str, disease: dict) -> dict:
        history = self._history[key]
        history.append(disease["probabilities"])

        labels = self.classifier.class_names
        averaged = {
            label: float(np.mean([item[label] for item in history]))
            for label in labels
        }
        best_label = max(averaged, key=averaged.get)
        return {
            "label": best_label,
            "confidence": averaged[best_label],
            "window": len(history),
            "probabilities": averaged,
        }

    def _pad_box(self, box: np.ndarray, image_w: int, image_h: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = [float(v) for v in box]
        bw = x2 - x1
        bh = y2 - y1
        pad = max(bw, bh) * self.crop_padding
        x1 = max(0, int(round(x1 - pad)))
        y1 = max(0, int(round(y1 - pad)))
        x2 = min(image_w, int(round(x2 + pad)))
        y2 = min(image_h, int(round(y2 + pad)))
        if x2 <= x1:
            x2 = min(image_w, x1 + 1)
        if y2 <= y1:
            y2 = min(image_h, y1 + 1)
        return x1, y1, x2, y2

    @staticmethod
    def _xyxy_to_norm_xywh(x1: int, y1: int, x2: int, y2: int, image_w: int, image_h: int) -> list[float]:
        width = x2 - x1
        height = y2 - y1
        return [
            (x1 + width / 2) / image_w,
            (y1 + height / 2) / image_h,
            width / image_w,
            height / image_h,
        ]
