import os, sys, shutil, glob, hashlib
from pathlib import Path
import kagglehub
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
from tqdm import tqdm

print("=== Step 1: Downloading Datasets via Kagglehub ===")

pepper_sources = [
    ("adithyantg/black-pepper-leaf-disease-mini-dataset", "mini"),
    ("udi17live/black-pepper-leaf-blight-and-yellow-mottle-virus", "blight_virus"),
    ("vijethajinu/black-pepper-dataset", "vijethajinu"),
]

pepper_raw_paths = {}
for repo_id, tag in pepper_sources:
    print(f"Downloading {repo_id}...")
    p = kagglehub.dataset_download(repo_id)
    pepper_raw_paths[tag] = p
    print(f" -> {tag}: {p}")

print("\nDownloading PlantVillage for Phase 1 Pretraining...")
pv_path = kagglehub.dataset_download("emmarex/plantdisease")
print(f" -> PlantVillage: {pv_path}")

print("\n=== Step 2: Setting up YOLO Segmentation Model on RTX 4090 ===")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
seg_model = YOLO("/root/models/leaf_detector_yolo26n_seg.pt")
print(f"YOLO Seg Model loaded on {device}")

def process_and_mask_crop(image_path, out_path, target_size=(320, 320)):
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return False
        h, w = img.shape[:2]
        
        # Run YOLO Seg inference
        results = seg_model.predict(img, imgsz=640, conf=0.25, device=device, verbose=False)
        result = results[0]
        
        if result.boxes is not None and len(result.boxes) > 0 and result.masks is not None and len(result.masks) > 0:
            # Pick the largest leaf mask by area
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            masks_data = result.masks.data.detach().cpu().numpy()
            
            best_idx = 0
            best_area = 0
            for idx, box in enumerate(boxes):
                area = (box[2] - box[0]) * (box[3] - box[1])
                if area > best_area:
                    best_area = area
                    best_idx = idx
            
            box = boxes[best_idx]
            mask_np = masks_data[best_idx]
            
            if mask_np.shape[:2] != (h, w):
                mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
            else:
                mask_resized = mask_np > 0.5
                
            # Apply neutral gray canvas
            masked_canvas = np.full_like(img, (128, 128, 128), dtype=np.uint8)
            masked_canvas[mask_resized] = img[mask_resized]
            
            # Pad box by 6%
            pad_w = int((box[2] - box[0]) * 0.06)
            pad_h = int((box[3] - box[1]) * 0.06)
            x1 = max(0, int(box[0]) - pad_w)
            y1 = max(0, int(box[1]) - pad_h)
            x2 = min(w, int(box[2]) + pad_w)
            y2 = min(h, int(box[3]) + pad_h)
            
            crop = masked_canvas[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                crop = img
        else:
            # Fallback: full image
            crop = img
            
        crop_resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), crop_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return True
    except Exception as e:
        return False

print("\n=== Step 3: Aggregating and Deduplicating Black Pepper Images ===")
pepper_classes = {
    "black_pepper_healthy": [],
    "black_pepper_footrot": [],
    "black_pepper_pollu_disease": [],
    "black_pepper_slow_decline": [],
    "black_pepper_yellow_mottle_virus": [],
    "black_pepper_leaf_blight": [],
}

seen_hashes = set()

for tag, root in pepper_raw_paths.items():
    for p in Path(root).glob("**/*.*"):
        if p.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        try:
            h = hashlib.md5(p.read_bytes()).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
        except:
            continue
            
        full_str = str(p).lower()
        if "footrot" in full_str or "foot_rot" in full_str:
            pepper_classes["black_pepper_footrot"].append(p)
        elif "pollu_disease" in full_str or "pollu" in full_str or "anthracnose" in full_str:
            pepper_classes["black_pepper_pollu_disease"].append(p)
        elif "slow-decline" in full_str or "slow_decline" in full_str:
            pepper_classes["black_pepper_slow_decline"].append(p)
        elif "yellow_mottle" in full_str:
            pepper_classes["black_pepper_yellow_mottle_virus"].append(p)
        elif "leaf_blight" in full_str or "blight" in full_str:
            pepper_classes["black_pepper_leaf_blight"].append(p)
        elif "healthy" in full_str:
            pepper_classes["black_pepper_healthy"].append(p)

print("Unique Black Pepper image counts:")
for c, imgs in pepper_classes.items():
    print(f" - {c}: {len(imgs)} images")

print("\n=== Step 4: Mask-Cropping and Splitting Black Pepper Target Dataset ===")
target_dataset_dir = Path("/root/data/black_pepper_target")
if target_dataset_dir.exists():
    shutil.rmtree(target_dataset_dir)

for c, imgs in pepper_classes.items():
    if len(imgs) == 0:
        continue
    train_imgs, test_val_imgs = train_test_split(imgs, test_size=0.20, random_state=42)
    val_imgs, test_imgs = train_test_split(test_val_imgs, test_size=0.50, random_state=42)
    
    splits = [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]
    for split_name, split_list in splits:
        print(f"Processing {c} -> {split_name} ({len(split_list)} images)...")
        for i, src_p in enumerate(tqdm(split_list, desc=f"{c}_{split_name}")):
            out_p = target_dataset_dir / split_name / c / f"{c}_{i:04d}.jpg"
            process_and_mask_crop(src_p, out_p)

print("\n=== Step 5: Preparing PlantVillage Pretrain Dataset (Tomato/Pepper/Potato) ===")
pv_pretrain_dir = Path("/root/data/plantvillage_pretrain")
if pv_pretrain_dir.exists():
    shutil.rmtree(pv_pretrain_dir)

pv_images = []
for p in Path(pv_path).glob("**/*.*"):
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        folder_name = p.parent.name
        # Select key Solanaceae crops and diseases
        if any(k in folder_name.lower() for k in ["tomato", "pepper", "potato"]):
            pv_images.append((folder_name, p))

print(f"Selected {len(pv_images)} images from PlantVillage for Phase 1 Pretraining.")
pv_by_class = {}
for cls_name, p in pv_images:
    pv_by_class.setdefault(cls_name, []).append(p)

for cls_name, imgs in pv_by_class.items():
    train_imgs, val_imgs = train_test_split(imgs, test_size=0.15, random_state=42)
    clean_cls = cls_name.replace("___", "_").replace(" ", "_")
    
    for split_name, split_list in [("train", train_imgs), ("val", val_imgs)]:
        for i, src_p in enumerate(split_list):
            out_p = pv_pretrain_dir / split_name / clean_cls / f"{clean_cls}_{i:04d}.jpg"
            out_p.parent.mkdir(parents=True, exist_ok=True)
            # Copy or resize directly
            img = cv2.imread(str(src_p))
            if img is not None:
                img_res = cv2.resize(img, (320, 320), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(out_p), img_res, [cv2.IMWRITE_JPEG_QUALITY, 90])

print("\n=== All Dataset Preparation Completed Successfully! ===")
