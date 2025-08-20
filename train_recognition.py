"""Training script for the PlantRecognitionModel.

This script provides a simple command-line interface for training the
recognition subsystem of BioViT3R. It loads a dataset of multi-view
plants, feeds images through the Vision Transformer backbone, and optimises
the classification head to predict species labels. The script outputs
training progress metrics and saves the best-performing model based on
validation accuracy.

Usage:

    python -m biovit3r.train_recognition \
        --train-dir /path/to/train_data \
        --val-dir /path/to/val_data \
        --num-classes 20 \
        --epochs 10 \
        --batch-size 4 \
        --lr 1e-4 \
        --output-model /path/to/save.pth

By default, the script assumes each sample contains at least one view in
the ``rgb`` subdirectory. It uses the first view per sample for training.
You can extend this to multi-view training by modifying the dataset and
training loop accordingly.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
from typing import Tuple

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import transforms
except ImportError as e:
    raise ImportError(
        "PyTorch is required to run the training script. "
        "Please install torch and torchvision."
    ) from e

from biovit3r.datasets import MultiViewPlantDataset
from biovit3r.models.recognition import PlantRecognitionModel
from sklearn.metrics import accuracy_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BioViT3R recognition model")
    parser.add_argument("--train-dir", type=str, required=True, help="Path to training data root")
    parser.add_argument("--val-dir", type=str, required=True, help="Path to validation data root")
    parser.add_argument("--num-classes", type=int, required=True, help="Number of species classes")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output-model", type=str, default="recognition_model.pth", help="Output model file")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze ViT backbone weights during training")
    return parser.parse_args()


def collate_fn(batch) -> Tuple[torch.Tensor, torch.Tensor]:
    # For recognition we only use the first RGB view per sample.
    images = []
    labels = []
    for rgb_views, _, _, label in batch:
        img = rgb_views[0]
        images.append(img)
        labels.append(label)
    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels


def main() -> None:
    args = parse_args()
    # Define basic transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_ds = MultiViewPlantDataset(args.train_dir, num_views=1, transform=transform)
    val_ds = MultiViewPlantDataset(args.val_dir, num_views=1, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PlantRecognitionModel(num_classes=args.num_classes)
    if args.freeze_backbone:
        model.freeze_backbone()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_samples = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * images.size(0)
            num_samples += images.size(0)
        epoch_loss /= max(num_samples, 1)
        # Evaluate
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                logits = model(images)
                preds = logits.argmax(dim=-1).cpu().numpy()
                val_preds.extend(preds.tolist())
                val_targets.extend(labels.numpy().tolist())
        val_acc = accuracy_score(val_targets, val_preds)
        print(f"Epoch {epoch}/{args.epochs} | Loss {epoch_loss:.4f} | Val acc {val_acc:.4f}")
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.output_model)
    print(f"Training complete. Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()