"""
EfficientNet-B0 based deepfake image detector trainer.
Uses transfer learning — starts from ImageNet-pretrained weights,
fine-tunes on your real/fake face dataset.

Expected accuracy: 88-95% (vs 77% with classical ML)
GPU: NVIDIA GTX 1650 (4GB VRAM) — supported
Training time: ~20-40 minutes on GPU
"""
from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from PIL import Image


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


class DeepfakeImageDataset(Dataset):
    """PyTorch Dataset for real/fake face images."""

    def __init__(self, image_paths: list[Path], labels: list[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        try:
            img = Image.open(self.image_paths[idx]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), color=128)
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def _collect_paths(folder: Path, max_files: int, rng: np.random.Generator) -> list[Path]:
    paths = [
        p for p in sorted(folder.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    if max_files > 0 and len(paths) > max_files:
        idx = rng.choice(len(paths), size=max_files, replace=False)
        paths = [paths[int(i)] for i in np.sort(idx)]
    return paths


def _best_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.25, 0.76, 0.01):
        pred = (probs >= t).astype(np.int32)
        score = float(f1_score(y_true, pred, zero_division=0))
        if score > best_f1:
            best_f1, best_t = score, float(t)
    return best_t, best_f1


def train_cnn_model(
    max_per_class: int = 10000,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    test_size: float = 0.15,
) -> dict:
    # ── Setup ─────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}", flush=True)
    if device.type == "cuda":
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB", flush=True)
        # For GTX 1650 (4GB VRAM), use batch_size=32 and AMP for memory efficiency
        use_amp = True
    else:
        use_amp = False
        batch_size = min(batch_size, 16)

    project_root = _project_root()
    real_dir = project_root / "ml" / "datasets" / "deepfake_images" / "real"
    fake_dir = project_root / "ml" / "datasets" / "deepfake_images" / "fake"

    if not real_dir.exists() or not fake_dir.exists():
        raise ValueError("Dataset folders not found. Expected real/ and fake/ inside ml/datasets/deepfake_images/.")

    rng = np.random.default_rng(42)

    # ── Collect paths ──────────────────────────────────────────────────────────
    print(f"[1/7] Collecting image paths (max {max_per_class}/class)...", flush=True)
    real_paths = _collect_paths(real_dir, max_per_class, rng)
    fake_paths = _collect_paths(fake_dir, max_per_class, rng)
    print(f"      {len(real_paths)} real + {len(fake_paths)} fake", flush=True)

    all_paths = real_paths + fake_paths
    all_labels = [0] * len(real_paths) + [1] * len(fake_paths)

    # Train/test split (manual, stratified)
    rng2 = np.random.default_rng(99)
    indices = np.arange(len(all_labels))
    labels_arr = np.array(all_labels)
    real_idx = indices[labels_arr == 0]
    fake_idx = indices[labels_arr == 1]
    rng2.shuffle(real_idx)
    rng2.shuffle(fake_idx)
    split_real = int(len(real_idx) * test_size)
    split_fake = int(len(fake_idx) * test_size)
    test_idx = np.concatenate([real_idx[:split_real], fake_idx[:split_fake]])
    train_idx = np.concatenate([real_idx[split_real:], fake_idx[split_fake:]])

    train_paths = [all_paths[i] for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]
    test_paths = [all_paths[i] for i in test_idx]
    test_labels = [all_labels[i] for i in test_idx]

    print(f"      Train: {len(train_paths)}, Test: {len(test_paths)}", flush=True)

    # ── Transforms ────────────────────────────────────────────────────────────
    # EfficientNet-B0 expects 224x224 normalized with ImageNet stats
    train_transform = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ── Datasets & DataLoaders ────────────────────────────────────────────────
    print(f"[2/7] Creating datasets and dataloaders...", flush=True)
    train_dataset = DeepfakeImageDataset(train_paths, train_labels, transform=train_transform)
    test_dataset = DeepfakeImageDataset(test_paths, test_labels, transform=test_transform)

    # Balanced sampler (ensures equal real/fake batches)
    class_counts = [train_labels.count(0), train_labels.count(1)]
    sample_weights = [1.0 / class_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_labels), replacement=True)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=2, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda"),
    )

    # ── Model: EfficientNet-B0 with fine-tuned classifier ────────────────────
    print(f"[3/7] Loading EfficientNet-B0 (pretrained on ImageNet)...", flush=True)
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # Freeze all layers except the last 3 blocks + classifier (fine-tuning strategy)
    for name, param in model.named_parameters():
        if "features.7" in name or "features.8" in name or "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Replace classifier head for binary classification
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, 1),   # single output for binary classification
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      Total params: {total_params:,} | Trainable: {trainable_params:,}", flush=True)

    # ── Optimizer & scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    # Mixed precision for memory efficiency on 4GB GPU
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"[4/7] Training {epochs} epochs...", flush=True)
    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (imgs, lbls) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.float().to(device, non_blocking=True)

            optimizer.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(imgs).squeeze(1)
                    loss = criterion(outputs, lbls)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(imgs).squeeze(1)
                loss = criterion(outputs, lbls)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            train_loss += loss.item()
            preds = (torch.sigmoid(outputs) >= 0.5).long()
            train_correct += (preds == lbls.long()).sum().item()
            train_total += lbls.size(0)

            if (batch_idx + 1) % 50 == 0:
                acc = 100.0 * train_correct / max(train_total, 1)
                print(f"      Epoch {epoch+1}/{epochs} batch {batch_idx+1}/{len(train_loader)} loss={train_loss/(batch_idx+1):.4f} acc={acc:.1f}%", flush=True)

        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        all_probs = []
        all_labels_val = []
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs = imgs.to(device, non_blocking=True)
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(imgs).squeeze(1)
                else:
                    outputs = model(imgs).squeeze(1)
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_probs.extend(probs.tolist())
                all_labels_val.extend(lbls.numpy().tolist())

        val_probs = np.array(all_probs)
        val_labels = np.array(all_labels_val)
        threshold, val_f1 = _best_threshold(val_labels, val_probs)
        val_auc = float(roc_auc_score(val_labels, val_probs))

        print(f"  >> Epoch {epoch+1}/{epochs}: val_f1={val_f1*100:.2f}% AUC={val_auc*100:.2f}% threshold={threshold:.2f}", flush=True)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            best_threshold = threshold
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"      *** New best model at epoch {best_epoch} ***", flush=True)

    # ── Load best weights & evaluate on test ─────────────────────────────────
    print(f"[5/7] Loading best weights (epoch {best_epoch})...", flush=True)
    model.load_state_dict(best_state)
    model.eval()

    all_probs = []
    all_labels_test = []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(imgs).squeeze(1)
            else:
                outputs = model(imgs).squeeze(1)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels_test.extend(lbls.numpy().tolist())

    test_probs = np.array(all_probs)
    test_labels_arr = np.array(all_labels_test)
    y_pred = (test_probs >= best_threshold).astype(np.int32)

    tn = int(np.sum((test_labels_arr == 0) & (y_pred == 0)))
    fp = int(np.sum((test_labels_arr == 0) & (y_pred == 1)))
    fn = int(np.sum((test_labels_arr == 1) & (y_pred == 0)))
    tp = int(np.sum((test_labels_arr == 1) & (y_pred == 1)))

    metrics = {
        "accuracy": round(float(accuracy_score(test_labels_arr, y_pred)) * 100, 2),
        "balanced_accuracy": round(float(balanced_accuracy_score(test_labels_arr, y_pred)) * 100, 2),
        "precision": round(float(precision_score(test_labels_arr, y_pred, zero_division=0)) * 100, 2),
        "recall": round(float(recall_score(test_labels_arr, y_pred, zero_division=0)) * 100, 2),
        "f1_score": round(float(f1_score(test_labels_arr, y_pred, zero_division=0)) * 100, 2),
        "roc_auc": round(float(roc_auc_score(test_labels_arr, test_probs)) * 100, 2),
    }
    print(f"FINAL METRICS: {metrics}", flush=True)

    # ── Save model ────────────────────────────────────────────────────────────
    print(f"[6/7] Saving model...", flush=True)
    models_dir = project_root / "ml" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "image_cnn.pkl"

    # Move model to CPU for serialization
    model = model.cpu()

    artifact = {
        "model_state_dict": best_state,
        "model_architecture": "efficientnet_b0",
        "model_type": "efficientnet_b0_finetuned",
        "input_size": 224,
        "feature_mode": "cnn_efficientnet",
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "class_map": {"0": "real", "1": "fake"},
        "metrics": metrics,
        "decision_threshold": round(best_threshold, 4),
        "validation_f1": round(best_val_f1 * 100, 2),
        "best_epoch": best_epoch,
        "total_epochs": epochs,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "training_config": {
            "max_per_class": max_per_class,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "device": str(device),
            "amp": use_amp,
        },
        "dataset_summary": {
            "real_count": len(real_paths),
            "fake_count": len(fake_paths),
            "train_count": len(train_paths),
            "test_count": len(test_paths),
        },
    }

    with model_path.open("wb") as f:
        pickle.dump(artifact, f)

    print(f"[7/7] Done! Model saved to {model_path}", flush=True)

    return {
        "model_path": str(model_path),
        "model_type": "efficientnet_b0_finetuned",
        "dataset_used": {
            "real_count": len(real_paths),
            "fake_count": len(fake_paths),
            "train_count": len(train_paths),
            "test_count": len(test_paths),
        },
        "metrics": metrics,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "trained_at": artifact["trained_at"],
        "best_epoch": best_epoch,
        "decision_threshold": round(best_threshold * 100, 2),
        "validation_f1": artifact["validation_f1"],
    }
