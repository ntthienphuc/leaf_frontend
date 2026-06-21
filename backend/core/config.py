from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


API_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.getenv("LEAF_MODELS_DIR", API_DIR / "core" / "models"))
TRACKERS_DIR = API_DIR / "trackers"


@dataclass(frozen=True)
class Settings:
    leaf_detector_path: Path = Path(
        os.getenv("LEAF_DETECTOR_PATH", MODELS_DIR / "leaf_detector_yolo26n_seg.pt")
    )
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
    detector_conf: float = float(os.getenv("LEAF_DETECTOR_CONF", "0.25"))
    detector_iou: float = float(os.getenv("LEAF_DETECTOR_IOU", "0.5"))
    classifier_imgsz: int = int(os.getenv("DISEASE_CLASSIFIER_IMGSZ", "320"))
    crop_padding: float = float(os.getenv("LEAF_CROP_PADDING", "0.06"))
    smoothing_window: int = int(os.getenv("LEAF_SMOOTHING_WINDOW", "7"))
    max_frame_side: int = int(os.getenv("LEAF_MAX_FRAME_SIDE", "1280"))


settings = Settings()
