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
        detector_conf: float = 0.45,
        detector_iou: float = 0.5,
        classifier_imgsz: int = 320,
        classifier_conf: float = 0.35,
        min_leaf_area_ratio: float = 0.002,
        max_leaf_area_ratio: float = 0.70,
        min_mask_box_fill_ratio: float = 0.36,
        max_leaf_aspect_ratio: float = 6.0,
        crop_padding: float = 0.08,
        smoothing_window: int = 7,
        classifier_interval: int = 3,
        classifier_motion_threshold: float = 0.08,
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
        self.classifier_conf = classifier_conf
        self.min_leaf_area_ratio = min_leaf_area_ratio
        self.max_leaf_area_ratio = max_leaf_area_ratio
        self.min_mask_box_fill_ratio = min_mask_box_fill_ratio
        self.max_leaf_aspect_ratio = max_leaf_aspect_ratio
        self.crop_padding = crop_padding
        self.smoothing_window = smoothing_window
        self.classifier_interval = max(1, int(classifier_interval))
        self.classifier_motion_threshold = max(0.0, float(classifier_motion_threshold))
        self._frame_index = 0
        self._classification_cache: dict[int, tuple[int, np.ndarray, dict]] = {}
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

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        use_tracker: bool = True,
        detector_conf: float | None = None,
        classifier_conf: float | None = None,
    ) -> dict:
        height, width = frame_bgr.shape[:2]

        det_threshold = self._confidence_threshold(detector_conf, self.detector_conf)
        cls_threshold = self._confidence_threshold(classifier_conf, self.classifier_conf)

        with self._lock:
            self._frame_index += 1
            if use_tracker:
                results = self.detector.track(
                    source=frame_bgr,
                    persist=True,
                    tracker=str(self.tracker_config_path),
                    imgsz=self.detector_imgsz,
                    conf=det_threshold,
                    iou=self.detector_iou,
                    device=self.device,
                    verbose=False,
                )
            else:
                results = self.detector.predict(
                    source=frame_bgr,
                    imgsz=self.detector_imgsz,
                    conf=det_threshold,
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
        det_scores = result.boxes.conf.detach().cpu().numpy()
        ids = result.boxes.id.detach().cpu().numpy().astype(int) if result.boxes.id is not None else None

        # Check if segment masks exist in result
        has_masks = result.masks is not None and len(result.masks) > 0
        masks_data = result.masks.data if has_masks else None
        masks_xy = result.masks.xy if has_masks else None

        for index, box in enumerate(boxes):
            x1, y1, x2, y2 = self._pad_box(box, width, height)
            
            # Apply binary mask with neutral gray fill (128, 128, 128)
            poly_list = None
            mask_resized = None
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

            if not self._passes_leaf_filters(box, mask_resized, width, height):
                continue

            # In case the crop is empty, skip
            if crop.size == 0:
                continue

            track_id: int | None = int(ids[index]) if ids is not None else None
            with self._lock:
                disease = self._predict_disease(crop, track_id, box, use_tracker)
                stable = (
                    self._smooth_prediction(track_id if track_id is not None else f"det-{index}", disease)
                    if use_tracker
                    else self._single_frame_prediction(disease)
                )

            # Filter out predictions below classification threshold
            if stable["confidence"] < cls_threshold:
                continue

            detections.append(
                {
                    "track_id": track_id,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "bbox_norm_xywh": self._xyxy_to_norm_xywh(x1, y1, x2, y2, width, height),
                    "mask_xy": poly_list,
                    "leaf_confidence": float(det_scores[index]),
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
            self._classification_cache.clear()
            self._frame_index = 0

    def _predict_disease(
        self,
        crop: np.ndarray,
        track_id: int | None,
        box: np.ndarray,
        use_tracker: bool,
    ) -> dict:
        """Reuse a tracked leaf's classifier result until it moves or expires."""
        if not use_tracker or track_id is None:
            return self.classifier.predict(crop)

        cached = self._classification_cache.get(track_id)
        should_refresh = cached is None
        if cached is not None:
            last_frame, last_box, _ = cached
            refresh_due = self._frame_index - last_frame >= self.classifier_interval
            previous_area = max(1.0, float((last_box[2] - last_box[0]) * (last_box[3] - last_box[1])))
            current_area = max(1.0, float((box[2] - box[0]) * (box[3] - box[1])))
            center_last = np.array([(last_box[0] + last_box[2]) / 2, (last_box[1] + last_box[3]) / 2])
            center_now = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            diagonal = max(1.0, float(np.hypot(box[2] - box[0], box[3] - box[1])))
            motion = float(np.linalg.norm(center_now - center_last) / diagonal)
            area_change = abs(current_area - previous_area) / previous_area
            should_refresh = refresh_due or motion > self.classifier_motion_threshold or area_change > self.classifier_motion_threshold

        if should_refresh:
            disease = self.classifier.predict(crop)
            self._classification_cache[track_id] = (self._frame_index, box.copy(), disease)
            return disease
        return cached[2]

    @staticmethod
    def _confidence_threshold(value: float | None, default: float) -> float:
        threshold = default if value is None else float(value)
        return min(0.99, max(0.01, threshold))

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

    @staticmethod
    def _single_frame_prediction(disease: dict) -> dict:
        return {
            "label": disease["label"],
            "confidence": disease["confidence"],
            "window": 1,
            "probabilities": disease["probabilities"],
        }

    def _pad_box(self, box: np.ndarray, image_w: int, image_h: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = [float(v) for v in box]
        bw = x2 - x1
        bh = y2 - y1
        pad_x = bw * self.crop_padding
        pad_y = bh * self.crop_padding
        x1 = max(0, int(round(x1 - pad_x)))
        y1 = max(0, int(round(y1 - pad_y)))
        x2 = min(image_w, int(round(x2 + pad_x)))
        y2 = min(image_h, int(round(y2 + pad_y)))
        if x2 <= x1:
            x2 = min(image_w, x1 + 1)
        if y2 <= y1:
            y2 = min(image_h, y1 + 1)
        return x1, y1, x2, y2

    def _passes_leaf_filters(
        self,
        box: np.ndarray,
        mask: np.ndarray | None,
        image_w: int,
        image_h: int,
    ) -> bool:
        x1, y1, x2, y2 = self._clip_box(box, image_w, image_h)
        box_w = x2 - x1
        box_h = y2 - y1
        if box_w <= 0 or box_h <= 0:
            return False

        frame_area = image_w * image_h
        box_area = box_w * box_h
        aspect_ratio = max(box_w / box_h, box_h / box_w)
        if aspect_ratio > self.max_leaf_aspect_ratio:
            return False

        if mask is not None:
            leaf_area = int(np.count_nonzero(mask[y1:y2, x1:x2]))
            fill_ratio = leaf_area / box_area
            if fill_ratio < self.min_mask_box_fill_ratio:
                return False
        else:
            leaf_area = box_area

        area_ratio = leaf_area / frame_area
        return self.min_leaf_area_ratio <= area_ratio <= self.max_leaf_area_ratio

    @staticmethod
    def _clip_box(box: np.ndarray, image_w: int, image_h: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = [float(v) for v in box]
        return (
            max(0, int(round(x1))),
            max(0, int(round(y1))),
            min(image_w, int(round(x2))),
            min(image_h, int(round(y2))),
        )

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
