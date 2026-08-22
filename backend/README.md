---
title: Leaf
emoji: 🍃
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Leaf Disease Real-Time API

FastAPI backend for real-time leaf detection, tracking, and disease prediction.

## WebSocket Endpoint

```text
wss://thienphuc12339-leaf.hf.space/ws/detect
```

## Recommended Frontend Flow

1. Capture camera frame on browser canvas.
2. Encode frame as JPEG blob, 640px on the long side.
3. Send JPEG bytes over WebSocket.
4. API returns JSON with leaf boxes, `track_id`, disease prediction, and smoothed disease prediction.
5. Frontend overlays boxes and labels.

## Detection Tuning

The backend uses YOLO confidence plus post-processing filters before disease classification. These can be tuned on Hugging Face Space environment variables without changing code:

```text
LEAF_DETECTOR_CONF=0.45
LEAF_TRACKER_DETECTION_CONF=0.10
DISEASE_CLASSIFIER_CONF=0.35
LEAF_MIN_AREA_RATIO=0.002
LEAF_MAX_AREA_RATIO=0.70
LEAF_MIN_MASK_BOX_FILL_RATIO=0.36
LEAF_MAX_ASPECT_RATIO=6.0
```

`LEAF_DETECTOR_CONF` controls YOLO confidence. The other `LEAF_*` filters reject detections that are too small, too large, or too thin. The current detector is a one-class rectangular-bbox model; the legacy segmentation checkpoint remains supported only as a fallback.

`LEAF_TRACKER_DETECTION_CONF` is deliberately lower than the display threshold. These weak detections are passed only to ByteTrack to preserve an existing ID through a short detection flicker; they are never shown or classified. `track_buffer: 45` keeps an ID for about three seconds at the frontend's 15 FPS cap.

The disease crop is expanded to a square before classification. `LEAF_CLASSIFIER_INTERVAL=3` reuses a tracked leaf's classifier result for up to two intermediate frames, while `LEAF_CLASSIFIER_MOTION_THRESHOLD=0.08` forces a refresh when the tracked crop moves or changes size substantially.

Device selection is automatic by default. With `LEAF_DEVICE=auto`, the backend uses the first CUDA GPU when the runtime exposes one and falls back to CPU otherwise. `LEAF_DEVICE=cuda:0`, `LEAF_DEVICE=0`, and `LEAF_DEVICE=cpu` are also accepted; an unavailable CUDA request safely falls back to CPU. Check `/health` to see the resolved device, CUDA runtime, and GPU name.

The Docker image installs CUDA-enabled PyTorch wheels. They remain CPU-compatible, so the same image works on a CPU Space and automatically uses a Hugging Face GPU after a hardware upgrade. Do not set `LEAF_DEVICE=cpu` unless you deliberately want to disable that GPU.

For the rented GPU server, use `LEAF_DETECTOR_IMGSZ=960` and keep the frontend capture limit at 960px when leaves are small in the camera view. The free CPU Hugging Face Space is suitable for smoke tests and still images, not low-latency live inference.

### JSON Response Format

```json
{
  "frame": {"width": 1280, "height": 720},
  "detections": [
    {
      "track_id": 1,
      "bbox_xyxy": [120, 80, 320, 360],
      "bbox_norm_xywh": [0.17, 0.31, 0.16, 0.39],
      "leaf_confidence": 0.91,
      "disease": {
        "label": "black_pepper_healthy",
        "confidence": 0.95
      },
      "stable_disease": {
        "label": "black_pepper_healthy",
        "confidence": 0.93,
        "window": 7
      }
    }
  ]
}
```
