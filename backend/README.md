# Leaf Disease Real-Time API

FastAPI backend for real-time leaf detection, tracking, and disease prediction.

Pipeline:

```text
camera frame
-> YOLO26 leaf detector
-> ByteTrack tracking
-> crop each leaf
-> EfficientNetV2-S disease classifier
-> JSON result to frontend
```

## Run

```powershell
cd D:\Project\Leaf\backend
.\run.ps1
```

API:

```text
http://localhost:8000
```

Health check:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

## WebSocket

Endpoint:

```text
ws://localhost:8000/ws/detect
```

Recommended frontend flow:

1. Capture camera frame on browser canvas.
2. Encode frame as JPEG blob, ideally 640-960 px on the long side.
3. Send JPEG bytes over WebSocket.
4. API returns JSON with leaf boxes, `track_id`, disease prediction, and smoothed disease prediction.
5. Frontend overlays boxes and labels locally.

Response shape:

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

## REST Test

```powershell
curl.exe -X POST "http://localhost:8000/predict/image" -F "file=@D:\path\to\image.jpg"
```

## Model Files

```text
core/models/leaf_detector_yolo26n.pt
core/models/disease_classifier_efficientnet_v2_s.pt
```

ONNX exports are included for later deployment, but the API loads PyTorch checkpoints to preserve class order and reduce integration mistakes.

## Frontend Recommendation

Use WebSocket for this task. It is simpler than WebRTC and works well when the browser only sends compressed JPEG frames and receives JSON boxes. Use WebRTC only if you need very low latency, multi-client streaming, or server-side media routing.
