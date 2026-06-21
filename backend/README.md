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
