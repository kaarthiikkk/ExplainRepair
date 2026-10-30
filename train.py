"""
Standalone script to train ResNet-18 on CIFAR-10.
Saves model checkpoint and training history.
"""

import argparse
import time
from pathlib import Path
import torch

from config import default_config, DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR
from models.model import build_model, load_cifar10_dataloaders, train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet-18 on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--train-subset", type=int, default=10000, help="Subset of training set to use (None for all 50k)")
    parser.add_argument("--val-subset", type=int, default=2000, help="Subset of validation set to use (None for all 10k)")
    parser.add_argument("--quick", action="store_true", help="Quick mode (1 epoch, 2000 train samples)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = default_config.device
    print(f"\n=================== TRAINING ResNet-18 on CIFAR-10 ===================")
    print(f"Hardware Accelerator: {device}")
    
    if args.quick:
        epochs = 1
        train_sub = 2000
        val_sub = 500
    else:
        epochs = args.epochs
        train_sub = args.train_subset
        val_sub = args.val_subset

    print(f"Configuration: {epochs} Epochs | Batch Size: {args.batch_size} | LR: {args.lr}")
    print(f"Data Samples: {train_sub} Train | {val_sub} Val")

    train_loader, val_loader = load_cifar10_dataloaders(
        data_dir=DATA_DIR,
        batch_size=args.batch_size,
        train_subset_size=train_sub,
        val_subset_size=val_sub
    )

    model, status = build_model(num_classes=10, checkpoint_path=None, device=device)
    checkpoint_path = default_config.model.checkpoint_path
    history_path = RESULTS_DIR / "training_history.json"

    def on_epoch(ep, tot, tl, ta, vl, va):
        print(f"  [Epoch {ep:02d}/{tot:02d}] Train Loss: {tl:.4f} | Train Acc: {ta*100:.2f}% | Val Loss: {vl:.4f} | Val Acc: {va*100:.2f}%")

    start_t = time.perf_counter()
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=args.lr,
        device=device,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        progress_callback=on_epoch
    )
    total_time = time.perf_counter() - start_t

    final_val_acc = history["val_acc"][-1] if history["val_acc"] else 0.0
    print(f"\nTraining completed in {total_time:.2f}s!")
    print(f"Final Validation Accuracy: {final_val_acc*100:.2f}%")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"History saved to: {history_path}")
    print("======================================================================\n")


if __name__ == "__main__":
    main()
