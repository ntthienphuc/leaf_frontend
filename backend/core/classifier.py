from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from .device import resolve_device


class DiseaseClassifier:
    def __init__(self, checkpoint_path: Path, device: str = "auto", image_size: int = 320) -> None:
        self.device = self._resolve_device(device)
        self.checkpoint_path = Path(checkpoint_path)
        
        # Optimize CPU threading for multi-core inference
        if self.device == "cpu":
            num_cores = os.cpu_count() or 4
            torch.set_num_threads(max(1, min(num_cores, 8)))

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        self.model_name = checkpoint.get("model_name", "efficientnet_v2_s")
        self.class_names = list(checkpoint["class_names"])
        self.image_size = int(checkpoint.get("image_size", checkpoint.get("img_size", image_size)))
        self.model = self._build_model(self.model_name, len(self.class_names))
        self.model.load_state_dict(checkpoint.get("state_dict", checkpoint.get("model_state")))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(int(self.image_size * 1.12), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @staticmethod
    def _resolve_device(device: str) -> str:
        return resolve_device(device)

    @staticmethod
    def _build_model(model_name: str, num_classes: int) -> nn.Module:
        if model_name == "efficientnet_v2_s":
            model = models.efficientnet_v2_s(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
            return model

        if model_name == "convnext_tiny":
            model = models.convnext_tiny(weights=None)
            in_features = model.classifier[2].in_features
            model.classifier[2] = nn.Linear(in_features, num_classes)
            return model

        raise ValueError(f"Unsupported classifier model: {model_name}")

    @torch.inference_mode()
    def predict(self, crop_bgr: np.ndarray) -> dict:
        """Single crop prediction (delegates to predict_batch)."""
        if crop_bgr.size == 0:
            raise ValueError("Empty crop passed to disease classifier")
        results = self.predict_batch([crop_bgr])
        return results[0]

    @torch.inference_mode()
    def predict_batch(self, crops_bgr: Sequence[np.ndarray]) -> list[dict]:
        """High-speed batch prediction for all leaves in a frame with 1 forward pass."""
        if not crops_bgr:
            return []

        tensors = []
        for crop in crops_bgr:
            if crop.size == 0:
                crop = np.full((self.image_size, self.image_size, 3), 128, dtype=np.uint8)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(crop_rgb))

        batch_tensor = torch.stack(tensors).to(self.device, non_blocking=True)

        if self.device.startswith("cuda"):
            with torch.amp.autocast(device_type="cuda"):
                logits = self.model(batch_tensor)
        else:
            logits = self.model(batch_tensor)

        probs_batch = torch.softmax(logits, dim=1).detach().cpu().numpy()
        results = []
        for probs in probs_batch:
            idx = int(np.argmax(probs))
            results.append(
                {
                    "label": self.class_names[idx],
                    "confidence": float(probs[idx]),
                    "probabilities": {
                        class_name: float(probs[i])
                        for i, class_name in enumerate(self.class_names)
                    },
                }
            )
        return results
