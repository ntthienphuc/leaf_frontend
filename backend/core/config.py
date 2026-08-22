from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


API_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.getenv("LEAF_MODELS_DIR", API_DIR / "core" / "models"))
TRACKERS_DIR = API_DIR / "trackers"


def default_detector_path() -> Path:
    """Prefer the leaf segmentation model."""
    configured = os.getenv("LEAF_DETECTOR_PATH")
    if configured:
        return Path(configured)
    seg_path = MODELS_DIR / "leaf_detector_yolo26n_seg.pt"
    detect_path = MODELS_DIR / "leaf_detector_yolo26n_detect.pt"
    return seg_path if seg_path.is_file() else detect_path


@dataclass(frozen=True)
class Settings:
    leaf_detector_path: Path = default_detector_path()
    disease_classifier_path: Path = Path(
        os.getenv(
            "DISEASE_CLASSIFIER_PATH",
            MODELS_DIR / "disease_classifier_efficientnet_v2_s_seg.pt",
        )
    )
    tracker_config_path: Path = Path(
        os.getenv("LEAF_TRACKER_CONFIG", TRACKERS_DIR / "bytetrack.yaml")
    )
    device: str = os.getenv("LEAF_DEVICE", "auto")
    detector_imgsz: int = int(os.getenv("LEAF_DETECTOR_IMGSZ", "640"))
    detector_conf: float = float(os.getenv("LEAF_DETECTOR_CONF", "0.45"))
    detector_iou: float = float(os.getenv("LEAF_DETECTOR_IOU", "0.5"))
    tracker_detection_conf: float = float(os.getenv("LEAF_TRACKER_DETECTION_CONF", "0.10"))
    reid_enabled: str = os.getenv("LEAF_REID_ENABLED", "auto")
    classifier_imgsz: int = int(os.getenv("DISEASE_CLASSIFIER_IMGSZ", "320"))
    classifier_conf: float = float(os.getenv("DISEASE_CLASSIFIER_CONF", "0.35"))
    min_leaf_area_ratio: float = float(os.getenv("LEAF_MIN_AREA_RATIO", "0.002"))
    max_leaf_area_ratio: float = float(os.getenv("LEAF_MAX_AREA_RATIO", "0.70"))
    min_mask_box_fill_ratio: float = float(os.getenv("LEAF_MIN_MASK_BOX_FILL_RATIO", "0.36"))
    max_leaf_aspect_ratio: float = float(os.getenv("LEAF_MAX_ASPECT_RATIO", "6.0"))
    crop_padding: float = float(os.getenv("LEAF_CROP_PADDING", "0.06"))
    smoothing_window: int = int(os.getenv("LEAF_SMOOTHING_WINDOW", "7"))
    classifier_interval: int = int(os.getenv("LEAF_CLASSIFIER_INTERVAL", "3"))
    classifier_motion_threshold: float = float(os.getenv("LEAF_CLASSIFIER_MOTION_THRESHOLD", "0.08"))
    max_frame_side: int = int(os.getenv("LEAF_MAX_FRAME_SIDE", "1280"))


settings = Settings()
