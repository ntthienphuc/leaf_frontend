from __future__ import annotations

import base64
import json
from functools import lru_cache

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
import os

from core.config import settings
from core.pipeline import LeafDiseasePipeline


app = FastAPI(title="Leaf Disease Real-Time API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_pipeline() -> LeafDiseasePipeline:
    return LeafDiseasePipeline(
        leaf_detector_path=settings.leaf_detector_path,
        disease_classifier_path=settings.disease_classifier_path,
        tracker_config_path=settings.tracker_config_path,
        device=settings.device,
        detector_imgsz=settings.detector_imgsz,
        detector_conf=settings.detector_conf,
        detector_iou=settings.detector_iou,
        classifier_imgsz=settings.classifier_imgsz,
        crop_padding=settings.crop_padding,
        smoothing_window=settings.smoothing_window,
    )


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Cannot decode image bytes")
    return resize_max_side(frame, settings.max_frame_side)


def resize_max_side(frame: np.ndarray, max_side: int) -> np.ndarray:
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return frame

    scale = max_side / longest
    new_size = (int(round(width * scale)), int(round(height * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


@app.on_event("startup")
def startup() -> None:
    # Load models once at startup instead of on the first frame.
    get_pipeline()


@app.get("/")
def read_root():
    # Check in the same directory first
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    # Check in child frontend directory (e.g. in Docker)
    child_path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    if os.path.exists(child_path):
        return FileResponse(child_path)
    # Check in sibling frontend directory (local dev)
    sibling_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    if os.path.exists(sibling_path):
        return FileResponse(sibling_path)
    return HTMLResponse("<html><body><h1>Leaf Disease API is running. index.html not found.</h1></body></html>")


@app.get("/health")
def health() -> dict:
    pipeline = get_pipeline()
    return {
        "ok": True,
        "device": pipeline.device,
        "detector": str(settings.leaf_detector_path),
        "classifier": str(settings.disease_classifier_path),
        "classes": pipeline.classifier.class_names,
    }


@app.get("/classes")
def classes() -> dict:
    return {"classes": get_pipeline().classifier.class_names}


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...), tracker: bool = False) -> dict:
    image_bytes = await file.read()
    frame = decode_image_bytes(image_bytes)
    return get_pipeline().process_frame(frame, use_tracker=tracker)


@app.post("/tracker/reset")
def reset_tracker() -> dict:
    get_pipeline().reset_tracker()
    return {"ok": True}


@app.websocket("/ws/detect")
async def detect_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    pipeline = get_pipeline()

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"] is not None:
                frame = decode_image_bytes(message["bytes"])
                result = pipeline.process_frame(frame, use_tracker=True)
                await websocket.send_json(result)
                continue

            if "text" not in message or message["text"] is None:
                continue

            payload = json.loads(message["text"])
            if payload.get("type") == "reset":
                pipeline.reset_tracker()
                await websocket.send_json({"ok": True, "type": "reset"})
                continue

            if payload.get("type") == "frame":
                data_url = payload["image"]
                if "," in data_url:
                    data_url = data_url.split(",", 1)[1]
                image_bytes = base64.b64decode(data_url)
                frame = decode_image_bytes(image_bytes)
                result = pipeline.process_frame(frame, use_tracker=payload.get("tracker", True))
                await websocket.send_json(result)
                continue

            await websocket.send_json({"ok": False, "error": "Unsupported websocket message"})

    except WebSocketDisconnect:
        return
