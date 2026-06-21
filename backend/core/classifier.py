from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms


class DiseaseClassifier:
    def __init__(self, checkpoint_path: Path, device: str = "auto", image_size: int = 320) -> None:
        self.device = self._resolve_device(device)
        self.checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        self.model_name = checkpoint.get("model_name", "efficientnet_v2_s")
        self.class_names = list(checkpoint["class_names"])
        self.image_size = int(checkpoint.get("img_size", image_size))
        self.model = self._build_model(self.model_name, len(self.class_names))
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(int(self.image_size * 1.12)),
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
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

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
        if crop_bgr.size == 0:
            raise ValueError("Empty crop passed to disease classifier")

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        idx = int(np.argmax(probs))

        return {
            "label": self.class_names[idx],
            "confidence": float(probs[idx]),
            "probabilities": {
                class_name: float(probs[i])
                for i, class_name in enumerate(self.class_names)
            },
        }
