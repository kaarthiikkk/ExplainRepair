"""
ResNet-18 Model Adaptation and Training Engine for CIFAR-10.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision.models import resnet18, ResNet18_Weights

from config import CIFAR10_CLASSES, default_config
from utils.image_utils import get_cifar10_transforms


class ResNet18CIFAR10(nn.Module):
    """
    ResNet-18 adapted for 32x32 CIFAR-10 classification.
    Replaces 7x7 conv1 and maxpool with 3x3 conv1 and Identity
    to preserve spatial feature resolution at 32x32 scale.
    """
    def __init__(self, num_classes: int = 10, pretrained_backbone: bool = False):
        super().__init__()
        if pretrained_backbone:
            # Load PyTorch weights and adapt layers
            base = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.backbone = base
            # Adapt conv1 for 32x32
            self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.backbone.maxpool = nn.Identity()
            self.backbone.fc = nn.Linear(512, num_classes)
        else:
            base = resnet18(weights=None)
            base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            base.maxpool = nn.Identity()
            base.fc = nn.Linear(512, num_classes)
            self.backbone = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_model(
    num_classes: int = 10,
    checkpoint_path: Optional[Path] = None,
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Build and return the ResNet-18 CIFAR-10 model.
    Returns (model, metadata_dict) where metadata_dict indicates training status.
    """
    if device is None:
        device = default_config.device

    model = ResNet18CIFAR10(num_classes=num_classes)
    status = {
        "is_trained": False,
        "checkpoint_found": False,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "source": "random_initialization"
    }

    if checkpoint_path and Path(checkpoint_path).exists():
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                model.load_state_dict(ckpt["state_dict"])
                status["val_acc"] = ckpt.get("val_acc", None)
                status["epoch"] = ckpt.get("epoch", None)
            else:
                model.load_state_dict(ckpt)
            status["is_trained"] = True
            status["checkpoint_found"] = True
            status["source"] = "checkpoint_file"
        except Exception as e:
            status["error"] = str(e)
            status["source"] = f"load_failed_fallback_uninitialized: {e}"

    model.to(device)
    model.eval()
    return model, status


def load_cifar10_dataloaders(
    data_dir: Path,
    batch_size: int = 64,
    num_workers: int = 0,
    train_subset_size: Optional[int] = None,
    val_subset_size: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Downloads and prepares CIFAR-10 train and test DataLoaders.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    train_transform = get_cifar10_transforms(train=True)
    val_transform = get_cifar10_transforms(train=False)

    train_set = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=True, download=True, transform=train_transform
    )
    val_set = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=False, download=True, transform=val_transform
    )

    if train_subset_size and train_subset_size < len(train_set):
        train_set = Subset(train_set, range(train_subset_size))
    if val_subset_size and val_subset_size < len(val_set):
        val_set = Subset(val_set, range(val_subset_size))

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 5,
    learning_rate: float = 0.001,
    device: Optional[torch.device] = None,
    checkpoint_path: Optional[Path] = None,
    history_path: Optional[Path] = None,
    progress_callback: Optional[Any] = None
) -> Dict[str, list]:
    """
    Train ResNet-18 on CIFAR-10 and save training history & best weights.
    """
    if device is None:
        device = default_config.device

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_train_loss = running_loss / max(1, total)
        epoch_train_acc = correct / max(1, total)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        epoch_val_loss = val_loss / max(1, val_total)
        epoch_val_acc = val_correct / max(1, val_total)
        scheduler.step()

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        if progress_callback:
            progress_callback(epoch, epochs, epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc)

        # Save best model
        if epoch_val_acc > best_val_acc and checkpoint_path:
            best_val_acc = epoch_val_acc
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "val_acc": epoch_val_acc,
                "history": history
            }, checkpoint_path)

    # Save final history
    if history_path:
        Path(history_path).parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    return history
