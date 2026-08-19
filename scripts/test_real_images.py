import os, sys, json
from pathlib import Path
import cv2
import requests
import numpy as np

# 1. Test live HF Space /health
print("=== 1. Testing Hugging Face Space Endpoint ===")
try:
    r = requests.get("https://thienphuc12339-leaf.hf.space/health", timeout=10)
    print("HF Space Health:", r.status_code, r.json())
except Exception as e:
    print("HF Space Health error:", e)

# 2. Test live HF Space /predict/image with real images
image_paths = [
    r"C:\Users\thien\.gemini\antigravity-ide\brain\921c37dc-0a5b-4102-8991-62d2a3f1f220\.user_uploaded\media_1787098556610.jpg",
    r"C:\Users\thien\.gemini\antigravity-ide\brain\921c37dc-0a5b-4102-8991-62d2a3f1f220\.user_uploaded\media_1787098556687.jpg",
    r"C:\Users\thien\.gemini\antigravity-ide\brain\921c37dc-0a5b-4102-8991-62d2a3f1f220\.user_uploaded\media_1787098556698.jpg",
]

for idx, p in enumerate(image_paths, 1):
    print(f"\n--- Testing Real Image #{idx}: {Path(p).name} ---")
    if not os.path.exists(p):
        print(f"File not found: {p}")
        continue
    
    with open(p, "rb") as f:
        img_bytes = f.read()
    
    # Send to HF Space
    try:
        r = requests.post(
            "https://thienphuc12339-leaf.hf.space/predict/image",
            files={"file": (Path(p).name, img_bytes, "image/jpeg")},
            params={"detector_conf": 0.25, "classifier_conf": 0.25},
            timeout=15
        )
        if r.status_code == 200:
            res = r.json()
            detections = res.get("detections", [])
            print(f"Status: 200 OK | Found {len(detections)} leaves in image")
            for d_i, d in enumerate(detections, 1):
                poly_len = len(d.get("polygon", []))
                print(f" Leaf {d_i}: Box={d.get('box')} | SegPolygon={poly_len} points | Label={d.get('disease_label')} | Conf={d.get('disease_confidence')*100:.1f}%")
        else:
            print(f"HF Space error {r.status_code}: {r.text}")
    except Exception as e:
        print("Request error:", e)
