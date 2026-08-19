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
                track_buffer=30,
                match_thresh=0.80,
                fuse_score=True,
            )
        return BYTETracker(args)

    def update_tracker(self, boxes: Any) -> np.ndarray:
        """Update the per-session tracker with detection boxes."""
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 8), dtype=np.float32)
        try:
            return self.tracker.update(boxes)
        except Exception:
            return np.empty((0, 8), dtype=np.float32)

    def reset(self) -> None:
        """Reset only this user's tracking and caching history."""
        self.tracker = self._build_tracker()
        self.frame_index = 0
        self.classification_cache.clear()
        self.history.clear()
