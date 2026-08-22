from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import uuid
import numpy as np
import yaml

from ultralytics.trackers.byte_tracker import BYTETracker


class UserSessionContext:
    """
    Encapsulates all per-client state for a WebSocket connection.
    Guarantees 100% isolation between multiple concurrent users sharing the same model weights.
    """

    def __init__(
        self,
        session_id: str | None = None,
        tracker_config_path: Path | str | None = None,
        smoothing_window: int = 7,
        classifier_interval: int = 3,
        classifier_motion_threshold: float = 0.08,
    ) -> None:
        self.session_id: str = session_id or str(uuid.uuid4())[:8]
        self.smoothing_window: int = max(1, int(smoothing_window))
        self.classifier_interval: int = max(1, int(classifier_interval))
        self.classifier_motion_threshold: float = max(0.0, float(classifier_motion_threshold))

        # Per-user ByteTrack tracker instance
        self.tracker_config_path = Path(tracker_config_path) if tracker_config_path else None
        self.tracker = self._build_tracker()

        # Per-user temporal queues & caches
        self.frame_index: int = 0
        self.classification_cache: dict[int, tuple[int, np.ndarray, dict]] = {}
        self.history: dict[int | str, deque[dict[str, float]]] = defaultdict(
            lambda: deque(maxlen=self.smoothing_window)
        )
        # ByteTrack IDs can swap when two same-class leaves overlap. Keep a
        # short spatial identity layer above it so classifier state follows the
        # physical leaf rather than a transient tracker assignment.
        self.stable_tracks: dict[int, dict[str, Any]] = {}
        self.next_stable_id: int = 1

        # Dynamic runtime thresholds (configured per-session from UI)
        self.detector_conf: float | None = None
        self.classifier_conf: float | None = None

    def _build_tracker(self) -> BYTETracker:
        if self.tracker_config_path and self.tracker_config_path.is_file():
            with open(self.tracker_config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            args = SimpleNamespace(**cfg)
        else:
            args = SimpleNamespace(
                tracker_type="bytetrack",
                track_high_thresh=0.25,
                track_low_thresh=0.10,
                new_track_thresh=0.25,
                track_buffer=45,
                match_thresh=0.80,
                fuse_score=True,
            )
        return BYTETracker(args)

    def update_tracker(self, boxes: Any) -> np.ndarray:
        """Update the per-session tracker with detection boxes."""
        if boxes is None:
            return np.empty((0, 8), dtype=np.float32)
        try:
            return self.tracker.update(boxes)
        except Exception:
            return np.empty((0, 8), dtype=np.float32)

    @staticmethod
    def _box_iou(left: np.ndarray, right: np.ndarray) -> float:
        x1 = max(float(left[0]), float(right[0]))
        y1 = max(float(left[1]), float(right[1]))
        x2 = min(float(left[2]), float(right[2]))
        y2 = min(float(left[3]), float(right[3]))
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        left_area = max(1.0, float(left[2] - left[0])) * max(1.0, float(left[3] - left[1]))
        right_area = max(1.0, float(right[2] - right[0])) * max(1.0, float(right[3] - right[1]))
        return intersection / max(1.0, left_area + right_area - intersection)

    def stabilize_track_ids(
        self,
        tracks: np.ndarray,
        embeddings_by_index: dict[int, np.ndarray] | None = None,
    ) -> dict[int, int]:
        """Map tracker rows to stable IDs using geometry and optional appearance."""
        embeddings_by_index = embeddings_by_index or {}
        if tracks is None or len(tracks) == 0:
            for state in self.stable_tracks.values():
                state["missed"] = int(state.get("missed", 0)) + 1
            self._prune_stable_tracks()
            return {}

        current: list[dict[str, Any]] = []
        for row in tracks:
            if len(row) < 8:
                continue
            current.append({
                "row": row,
                "box": np.asarray(row[:4], dtype=np.float32),
                "raw_id": int(row[4]),
                "orig_idx": int(row[7]),
                "embedding": embeddings_by_index.get(int(row[7])),
            })

        # Greedy highest-quality matching is sufficient for the small number of
        # leaves in a camera frame and avoids a scipy dependency.
        matches: list[tuple[float, int, int]] = []
        spatial_candidate_indices: set[int] = set()
        for current_index, item in enumerate(current):
            box = item["box"]
            current_center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            current_diag = max(1.0, float(np.hypot(box[2] - box[0], box[3] - box[1])))
            for stable_id, state in self.stable_tracks.items():
                previous = state["box"]
                previous_center = np.array([(previous[0] + previous[2]) / 2, (previous[1] + previous[3]) / 2])
                distance = float(np.linalg.norm(current_center - previous_center) / current_diag)
                iou = self._box_iou(box, previous)
                area = max(1.0, float((box[2] - box[0]) * (box[3] - box[1])))
                previous_area = max(1.0, float((previous[2] - previous[0]) * (previous[3] - previous[1])))
                area_change = abs(np.log(area / previous_area))
                score = iou - 0.30 * min(distance, 3.0) - 0.08 * min(area_change, 3.0)
                embedding = item["embedding"]
                previous_embedding = state.get("embedding")
                if embedding is not None and previous_embedding is not None:
                    # Embeddings are L2 normalized, so dot product is cosine similarity.
                    appearance = float(np.clip(np.dot(embedding, previous_embedding), -1.0, 1.0))
                    score += 0.55 * appearance
                if iou >= 0.02 or distance <= 1.25:
                    matches.append((score, current_index, stable_id))
                    spatial_candidate_indices.add(current_index)

        # A rapid camera pan can move every leaf beyond the spatial gate. In
        # that case, retain ByteTrack's raw association as a fallback only;
        # spatial matches always take precedence when leaves are close enough
        # to be confused with one another.
        for current_index, item in enumerate(current):
            if current_index in spatial_candidate_indices:
                continue
            for stable_id, state in self.stable_tracks.items():
                if int(state.get("raw_id", -1)) == item["raw_id"]:
                    matches.append((-10.0, current_index, stable_id))

        matches.sort(reverse=True)
        assigned_current: set[int] = set()
        assigned_stable: set[int] = set()
        current_to_stable: dict[int, int] = {}
        for _, current_index, stable_id in matches:
            if current_index in assigned_current or stable_id in assigned_stable:
                continue
            assigned_current.add(current_index)
            assigned_stable.add(stable_id)
            current_to_stable[current_index] = stable_id

        for current_index, item in enumerate(current):
            stable_id = current_to_stable.get(current_index)
            if stable_id is None:
                stable_id = self.next_stable_id
                self.next_stable_id += 1
            item["stable_id"] = stable_id
            previous_state = self.stable_tracks.get(stable_id, {})
            self.stable_tracks[stable_id] = {
                "box": item["box"].copy(),
                "raw_id": item["raw_id"],
                "embedding": item["embedding"] if item["embedding"] is not None else previous_state.get("embedding"),
                "missed": 0,
            }

        for stable_id, state in self.stable_tracks.items():
            if stable_id not in current_to_stable.values() and stable_id not in {item["stable_id"] for item in current}:
                state["missed"] = int(state.get("missed", 0)) + 1
        self._prune_stable_tracks()
        return {item["orig_idx"]: item["stable_id"] for item in current}

    def _prune_stable_tracks(self) -> None:
        max_missed = int(getattr(self.tracker, "max_frames_lost", 45))
        expired = [sid for sid, state in self.stable_tracks.items() if int(state.get("missed", 0)) > max_missed]
        for stable_id in expired:
            self.stable_tracks.pop(stable_id, None)

    def reset(self) -> None:
        """Reset only this user's tracking and caching history."""
        self.tracker = self._build_tracker()
        self.frame_index = 0
        self.classification_cache.clear()
        self.history.clear()
        self.stable_tracks.clear()
        self.next_stable_id = 1
