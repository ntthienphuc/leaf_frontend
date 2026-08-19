import os, sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
import timm
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using compute device: {device} ({torch.cuda.get_device_name(0)})")

IMAGE_SIZE = 320
BATCH_SIZE = 64
NUM_WORKERS = 8

# Enhanced Augmentations
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=180),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def create_model(model_name: str, num_classes: int, pretrained: bool = True):
    if model_name == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
    elif model_name == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        # Timm fallback
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model.to(device)

def train_epoch(model, dataloader, criterion, optimizer, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def eval_epoch(model, dataloader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        with torch.amp.autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)

# =========================================================================
# STAGE 1: PlantVillage General Pathology Pre-training
# =========================================================================
print("\n" + "="*80)
print("STAGE 1: Universal Plant Pathology Pre-training on PlantVillage (~15k images)")
print("="*80)

pv_dir = Path("/root/data/plantvillage_pretrain")
pv_train_ds = datasets.ImageFolder(str(pv_dir / "train"), transform=train_transforms)
pv_val_ds = datasets.ImageFolder(str(pv_dir / "val"), transform=eval_transforms)
pv_train_loader = DataLoader(pv_train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
pv_val_loader = DataLoader(pv_val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

print(f"PlantVillage Pretrain Classes ({len(pv_train_ds.classes)}): {pv_train_ds.classes}")
print(f"Train samples: {len(pv_train_ds)} | Val samples: {len(pv_val_ds)}")

pv_models = {}
for m_name in ["convnext_tiny", "efficientnet_v2_s"]:
    print(f"\n--- Pretraining {m_name} on PlantVillage (5 Epochs) ---")
    model = create_model(m_name, num_classes=len(pv_train_ds.classes), pretrained=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    scaler = torch.amp.GradScaler("cuda")
    
    for ep in range(1, 6):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, pv_train_loader, criterion, optimizer, scaler)
        val_loss, val_acc, _, _ = eval_epoch(model, pv_val_loader, criterion)
        scheduler.step()
        print(f"Epoch {ep}/5 ({time.time()-t0:.1f}s) | Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%")
    
    pv_models[m_name] = model

# =========================================================================
# STAGE 2: Black Pepper Specialized Target Fine-Tuning
# =========================================================================
print("\n" + "="*80)
print("STAGE 2: Black Pepper Specialized Target Fine-Tuning (6 Classes)")
print("="*80)

target_dir = Path("/root/data/black_pepper_target")
train_ds = datasets.ImageFolder(str(target_dir / "train"), transform=train_transforms)
val_ds = datasets.ImageFolder(str(target_dir / "val"), transform=eval_transforms)
test_ds = datasets.ImageFolder(str(target_dir / "test"), transform=eval_transforms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

class_names = train_ds.classes
num_classes = len(class_names)
print(f"Target Black Pepper Classes ({num_classes}): {class_names}")
print(f"Target Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

out_models_dir = Path("/root/runs/classifier_experiments")
out_models_dir.mkdir(parents=True, exist_ok=True)
workspace_dir = Path("/workspace/models")
workspace_dir.mkdir(parents=True, exist_ok=True)

results_summary = []

for m_name in ["convnext_tiny", "efficientnet_v2_s"]:
    print(f"\n========================================================")
    print(f"Fine-Tuning {m_name} on Black Pepper Dataset (20 Epochs)")
    print(f"========================================================")
    
    # Load Stage 1 pretrained backbone and replace head for 6 target classes
    model = pv_models[m_name]
    if m_name == "convnext_tiny":
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes).to(device)
    elif m_name == "efficientnet_v2_s":
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes).to(device)
        
    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_weights = None
    best_epoch = 0
    patience = 8
    patience_counter = 0
    history = []
    
    for ep in range(1, 21):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc, _, _ = eval_epoch(model, val_loader, criterion)
        scheduler.step()
        
        history.append({
            "epoch": ep, "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": val_loss, "val_acc": val_acc
        })
        
        is_best = val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss)
        if is_best:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = ep
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            star = " ⭐ (Best)"
        else:
            patience_counter += 1
            star = ""
            
        print(f"Epoch {ep:2d}/20 ({time.time()-t0:.1f}s) | Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%{star}")
        
        if patience_counter >= patience:
            print(f"Early stopping triggered at Epoch {ep}!")
            break
            
    # Load best weights for final test evaluation
    model.load_state_dict(best_weights)
    test_loss, test_acc, test_preds, test_labels = eval_epoch(model, test_loader, criterion)
    test_f1 = f1_score(test_labels, test_preds, average="macro")
    
    print(f"\n--- Final Test Results for {m_name} ---")
    print(f"Best Val Acc: {best_val_acc*100:.2f}% (Epoch {best_epoch})")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test Macro F1: {test_f1*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=class_names, digits=4))
    
    # Save Confusion Matrix Plot
    cm = confusion_matrix(test_labels, test_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=class_names, yticklabels=class_names, ax=ax)
    plt.title(f"Confusion Matrix - {m_name} (Test Acc: {test_acc*100:.2f}%)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    cm_path = out_models_dir / f"{m_name}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    
    # Save Checkpoint with metadata
    ckpt = {
        "model_name": m_name,
        "class_names": class_names,
        "image_size": IMAGE_SIZE,
        "best_epoch": best_epoch,
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "test_f1": float(test_f1),
        "state_dict": best_weights,
    }
    
    ckpt_path = out_models_dir / f"disease_classifier_{m_name}_2stage.pt"
    torch.save(ckpt, ckpt_path)
    
    # Also backup to persistent network volume
    backup_path = workspace_dir / f"disease_classifier_{m_name}_2stage.pt"
    torch.save(ckpt, backup_path)
    print(f"Model saved to {ckpt_path} and backed up to {backup_path}")
    
    results_summary.append({
        "model_name": m_name,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "checkpoint": str(ckpt_path),
    })

# Summary table
df_summary = pd.DataFrame(results_summary)
print("\n" + "="*80)
print("FINAL BENCHMARK SUMMARY")
print("="*80)
print(df_summary.to_string(index=False))
df_summary.to_csv(out_models_dir / "classifier_benchmark_summary.csv", index=False)
df_summary.to_csv(workspace_dir / "classifier_benchmark_summary.csv", index=False)
print("\nAll 2-Stage Training Experiments Completed Successfully!")
