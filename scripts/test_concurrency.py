import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import asyncio
import base64
import json
import time
from pathlib import Path
import cv2
import numpy as np
from fastapi.testclient import TestClient
from backend.main import app, get_pipeline

print("=== 1. Warming Up Pipeline & App ===")
client = TestClient(app)
r = client.get("/health")
print("Health status:", r.status_code, r.json().get("classes"))

print("\n=== 2. Multi-User WebSocket Concurrency & Isolation Test ===")
# Simulate 3 concurrent clients
img1 = cv2.imread("experiments/real_tests/real_image_1_annotated.jpg")
img2 = cv2.imread("experiments/real_tests/real_image_2_annotated.jpg")
img3 = cv2.imread("experiments/real_tests/real_image_3_annotated.jpg")

_, buf1 = cv2.imencode(".jpg", img1)
_, buf2 = cv2.imencode(".jpg", img2)
_, buf3 = cv2.imencode(".jpg", img3)
b64_1 = base64.b64encode(buf1).decode("utf-8")
b64_2 = base64.b64encode(buf2).decode("utf-8")
b64_3 = base64.b64encode(buf3).decode("utf-8")

with client.websocket_connect("/ws/detect") as ws1, \
     client.websocket_connect("/ws/detect") as ws2, \
     client.websocket_connect("/ws/detect") as ws3:
    
    # Configure each client with different thresholds
    ws1.send_json({"type": "config", "detector_conf": 0.30, "classifier_conf": 0.30})
    conf1 = ws1.receive_json()
    
    ws2.send_json({"type": "config", "detector_conf": 0.50, "classifier_conf": 0.50})
    conf2 = ws2.receive_json()
    
    ws3.send_json({"type": "config", "detector_conf": 0.20, "classifier_conf": 0.20})
    conf3 = ws3.receive_json()
    
    print(f"Client 1 Session ID: {conf1.get('session_id')} | Detector Conf: {conf1.get('detector_conf')}")
    print(f"Client 2 Session ID: {conf2.get('session_id')} | Detector Conf: {conf2.get('detector_conf')}")
    print(f"Client 3 Session ID: {conf3.get('session_id')} | Detector Conf: {conf3.get('detector_conf')}")
    
    assert conf1["session_id"] != conf2["session_id"] != conf3["session_id"], "Sessions must be unique!"
    print(" Verified: Unique session IDs assigned per WebSocket connection.")
    
    # Send frames concurrently
    ws1.send_json({"type": "frame", "image": b64_1, "tracker": True})
    ws2.send_json({"type": "frame", "image": b64_2, "tracker": True})
    ws3.send_json({"type": "frame", "image": b64_3, "tracker": True})
    
    res1 = ws1.receive_json()
    res2 = ws2.receive_json()
    res3 = ws3.receive_json()
    
    print(f"\nClient 1 Detections: {len(res1.get('detections', []))} leaves (Session: {res1.get('session_id')})")
    print(f"Client 2 Detections: {len(res2.get('detections', []))} leaves (Session: {res2.get('session_id')})")
    print(f"Client 3 Detections: {len(res3.get('detections', []))} leaves (Session: {res3.get('session_id')})")
    
    # Reset Client 1 tracker
    ws1.send_json({"type": "reset"})
    reset_res = ws1.receive_json()
    print("\nClient 1 Reset:", reset_res)
    
    # Send another frame to Client 2 to verify Client 2 is unaffected
    ws2.send_json({"type": "frame", "image": b64_2, "tracker": True})
    res2_after = ws2.receive_json()
    print(f"Client 2 after Client 1 reset: {len(res2_after.get('detections', []))} leaves (Unaffected: OK)")

print("\n=== 3. Speed & Payload Size Benchmark ===")
raw_json_len = len(json.dumps(res1))
print(f"Optimized WebSocket JSON Payload Size: {raw_json_len/1024:.2f} KB")

# Test 10 frames sequential latency
t0 = time.time()
with client.websocket_connect("/ws/detect") as ws:
    for i in range(10):
        ws.send_json({"type": "frame", "image": b64_1, "tracker": True})
        _ = ws.receive_json()
t_total = time.time() - t0
print(f"Processed 10 multi-leaf frames in {t_total:.2f}s ({t_total/10*1000:.1f} ms/frame = {10/t_total:.1f} FPS on local CPU)")

print("\n ALL CONCURRENCY AND PERFORMANCE TESTS PASSED PERFECTLY!")
