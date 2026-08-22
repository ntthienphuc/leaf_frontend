from __future__ import annotations

from pathlib import Path
from typing import Sequence
import threading

import cv2
import numpy as np
from ultralytics import YOLO

from .classifier import DiseaseClassifier
from .device import resolve_device
from .session import UserSessionContext


class LeafDiseasePipeline:
    """
    High-Performance Shared Inference Engine for Leaf Detection & Disease Classification.
    Supports Multi-User Concurrency through UserSessionContext isolation.
    """

    def __init__(
        self,
        leaf_detector_path: Path,
        disease_classifier_path: Path,
        tracker_config_path: Path | None = None,
        device: str = "auto",
        detector_imgsz: int = 640,
        detector_conf: float = 0.45,
        detector_iou: float = 0.5,
        tracker_detection_conf: float = 0.10,
        classifier_imgsz: int = 320,
        classifier_conf: float = 0.35,
        min_leaf_area_ratio: float = 0.002,
        max_leaf_area_ratio: float = 0.70,
        min_mask_box_fill_ratio: float = 0.36,
        max_leaf_aspect_ratio: float = 6.0,
        crop_padding: float = 0.06,
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
        self.tracker_config_path = tracker_config_path
        self.detector_imgsz = detector_imgsz
        self.detector_conf = detector_conf
        self.detector_iou = detector_iou
        self.tracker_detection_conf = self._confidence_threshold(tracker_detection_conf, 0.10)
        self.classifier_conf = classifier_conf
        self.min_leaf_area_ratio = min_leaf_area_ratio
        self.max_leaf_area_ratio = max_leaf_area_ratio
        self.min_mask_box_fill_ratio = min_mask_box_fill_ratio
        self.max_leaf_aspect_ratio = max_leaf_aspect_ratio
        self.crop_padding = crop_padding
        self.smoothing_window = smoothing_window
        self.classifier_interval = max(1, int(classifier_interval))
        self.classifier_motion_threshold = max(0.0, float(classifier_motion_threshold))
        self._predict_lock = threading.Lock()

        # Warm up models once at init time to initialize weights & backends
        _dummy = np.zeros((self.detector_imgsz, self.detector_imgsz, 3), dtype=np.uint8)
        try:
            self.detector.predict(source=_dummy, imgsz=self.detector_imgsz, device=self.device, verbose=False)
            self.classifier.predict(_dummy[:self.classifier.image_size, :self.classifier.image_size])
        except Exception:
            pass

        # Default session for stateless non-WebSocket requests
        self._default_session = UserSessionContext(
            session_id="default",
            tracker_config_path=self.tracker_config_path,
            smoothing_window=self.smoothing_window,
            classifier_interval=self.classifier_interval,
            classifier_motion_threshold=self.classifier_motion_threshold,
        )

    @staticmethod
    def _resolve_device(device: str) -> str:
        return resolve_device(device)

    def create_session(self, session_id: str | None = None) -> UserSessionContext:
        """Create an isolated session context for a new WebSocket user."""
        return UserSessionContext(
            session_id=session_id,
            tracker_config_path=self.tracker_config_path,
            smoothing_window=self.smoothing_window,
            classifier_interval=self.classifier_interval,
            classifier_motion_threshold=self.classifier_motion_threshold,
        )

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        session: UserSessionContext | None = None,
        use_tracker: bool = True,
        detector_conf: float | None = None,
        classifier_conf: float | None = None,
    ) -> dict:
        """
        Process a single video/image frame with batched inference & session isolation.
        """
        ctx = session or self._default_session
        height, width = frame_bgr.shape[:2]

        det_threshold = self._confidence_threshold(
            detector_conf or ctx.detector_conf, self.detector_conf
        )
        cls_threshold = self._confidence_threshold(
            classifier_conf or ctx.classifier_conf, self.classifier_conf
        )

        ctx.frame_index += 1

        # Run YOLO below the UI threshold so ByteTrack can bridge brief weak detections.
        inference_conf = min(det_threshold, self.tracker_detection_conf) if use_tracker else det_threshold
        with self._predict_lock:
            results = self.detector.predict(
                source=frame_bgr,
                imgsz=self.detector_imgsz,
                conf=inference_conf,
                iou=self.detector_iou,
                device=self.device,
                verbose=False,
            )

        result = results[0]
        detections = []

        # Update even on empty frames. ByteTrack's lost-track timeout is measured
        # in processed frames, so skipping this would make its lifecycle inconsistent.
        track_id_map: dict[int, int] = {}
        if use_tracker:
            tracks = ctx.update_tracker(result.boxes)
            if len(tracks) > 0:
                for row in tracks:
                    # row format: [x1, y1, x2, y2, track_id, conf, cls, orig_idx]
                    t_id = int(row[4])
                    orig_idx = int(row[7]) if len(row) > 7 else -1
                    if orig_idx >= 0:
                        track_id_map[orig_idx] = t_id

        if result.boxes is None or len(result.boxes) == 0:
            return {
                "frame": {"width": width, "height": height},
                "detections": detections,
            }

        boxes = result.boxes.xyxy.detach().cpu().numpy()
        det_scores = result.boxes.conf.detach().cpu().numpy()

        # 3. Mask extraction & Douglas-Peucker Polygon Simplification
        has_masks = result.masks is not None and len(result.masks) > 0
        masks_data = result.masks.data if has_masks else None
        masks_xy = result.masks.xy if has_masks else None

        candidate_leaves = []
        for index, box in enumerate(boxes):
            # Weak detections are tracker-only; never show or classify them.
            if float(det_scores[index]) < det_threshold:
                continue
            x1, y1, x2, y2 = self._pad_box(box, width, height)
            poly_list = None
            mask_resized = None

            if has_masks and masks_data is not None and index < len(masks_data):
                mask_np = masks_data[index].detach().cpu().numpy()
                if mask_np.shape[:2] != (height, width):
                    mask_resized = cv2.resize(mask_np, (width, height), interpolation=cv2.INTER_NEAREST) > 0.5
                else:
                    mask_resized = mask_np > 0.5

                masked_canvas = np.full_like(frame_bgr, (128, 128, 128), dtype=np.uint8)
                masked_canvas[mask_resized] = frame_bgr[mask_resized]
                crop = masked_canvas[y1:y2, x1:x2]

                if masks_xy is not None and index < len(masks_xy):
                    raw_poly = np.array(masks_xy[index], dtype=np.float32)
                    if len(raw_poly) >= 3:
                        # Simplify polygon to reduce JSON payload by ~85%
                        epsilon = 1.2
                        simplified = cv2.approxPolyDP(raw_poly.reshape((-1, 1, 2)), epsilon, True)
                        poly_list = [[round(float(pt[0][0]), 1), round(float(pt[0][1]), 1)] for pt in simplified]
            else:
                crop = frame_bgr[y1:y2, x1:x2]

            if not self._passes_leaf_filters(box, mask_resized, width, height):
                continue
            if crop.size == 0:
                continue

            track_id = track_id_map.get(index)
            candidate_leaves.append({
                "index": index,
                "box": box,
                "bbox_xyxy": [x1, y1, x2, y2],
                "poly_list": poly_list,
                "leaf_conf": float(det_scores[index]),
                "track_id": track_id,
                "crop": crop,
            })

        if not candidate_leaves:
            return {
                "frame": {"width": width, "height": height},
                "detections": detections,
            }

        # 4. Batch Classification with Intelligent Spatial Cache
        crops_to_classify = []
        classify_indices = []

        for item_idx, leaf in enumerate(candidate_leaves):
            track_id = leaf["track_id"]
            box = leaf["box"]
            should_refresh = True

            if use_tracker and track_id is not None:
                cached = ctx.classification_cache.get(track_id)
                if cached is not None:
                    last_frame, last_box, cached_disease = cached
                    refresh_due = (ctx.frame_index - last_frame) >= ctx.classifier_interval
                    prev_area = max(1.0, float((last_box[2] - last_box[0]) * (last_box[3] - last_box[1])))
                    curr_area = max(1.0, float((box[2] - box[0]) * (box[3] - box[1])))
                    c_last = np.array([(last_box[0] + last_box[2]) / 2, (last_box[1] + last_box[3]) / 2])
                    c_now = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
                    diag = max(1.0, float(np.hypot(box[2] - box[0], box[3] - box[1])))
                    motion = float(np.linalg.norm(c_now - c_last) / diag)
                    area_change = abs(curr_area - prev_area) / prev_area

                    if not (refresh_due or motion > ctx.classifier_motion_threshold or area_change > ctx.classifier_motion_threshold):
                        should_refresh = False
                        leaf["disease"] = cached_disease

            if should_refresh:
                crops_to_classify.append(leaf["crop"])
                classify_indices.append(item_idx)

        # Single batched forward pass for all leaves needing prediction
        if crops_to_classify:
            with self._predict_lock:
                batch_predictions = self.classifier.predict_batch(crops_to_classify)
            for idx, pred in zip(classify_indices, batch_predictions):
                candidate_leaves[idx]["disease"] = pred
                track_id = candidate_leaves[idx]["track_id"]
                if use_tracker and track_id is not None:
                    ctx.classification_cache[track_id] = (
                        ctx.frame_index,
                        candidate_leaves[idx]["box"].copy(),
                        pred,
                    )

        # 5. Assemble final detections with temporal smoothing
        for leaf in candidate_leaves:
            disease = leaf.get("disease")
            if disease is None:
                continue

            track_id = leaf["track_id"]
            if use_tracker:
                key = track_id if track_id is not None else f"det-{leaf['index']}"
                stable = self._smooth_prediction(ctx, key, disease)
            else:
                stable = self._single_frame_prediction(disease)

            if stable["confidence"] < cls_threshold:
                continue

            x1, y1, x2, y2 = leaf["bbox_xyxy"]
            detections.append(
                {
                    "track_id": track_id,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "bbox_norm_xywh": self._xyxy_to_norm_xywh(x1, y1, x2, y2, width, height),
                    "mask_xy": leaf["poly_list"],
                    "leaf_confidence": leaf["leaf_conf"],
                    "disease": disease,
                    "stable_disease": stable,
                }
            )

        return {
            "frame": {"width": width, "height": height},
            "detections": detections,
        }

    def reset_tracker(self, session: UserSessionContext | None = None) -> None:
        """Reset tracker for a specific session (or default session)."""
        ctx = session or self._default_session
        ctx.reset()

    @staticmethod
    def _confidence_threshold(value: float | None, default: float) -> float:
        threshold = default if value is None else float(value)
        return min(0.99, max(0.01, threshold))

    def _smooth_prediction(self, ctx: UserSessionContext, key: int | str, disease: dict) -> dict:
        history = ctx.history[key]
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
            round((x1 + width / 2) / image_w, 4),
            round((y1 + height / 2) / image_h, 4),
            round(width / image_w, 4),
            round(height / image_h, 4),
        ]
