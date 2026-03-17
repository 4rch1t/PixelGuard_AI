"""Train a binary classifier to distinguish REAL vs AI-generated (FAKE) images.

Assumes dataset layout:

    train/
      REAL/
      FAKE/

This script uses the same ResNet6Ch model (RGB+ELA) as used by app.py.

Example:
    python train_real_vs_ai.py --data-dir train --out-dir runs/real_vs_ai --epochs 10
"""

import argparse
import math
import os
import random
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Reuse helpers from the existing train_casia2.py so the model matches the app.
from train_casia2 import ResNet6Ch, compute_ela_rgb


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RealFakeDataset(Dataset):
    """Dataset for images stored under train/REAL and train/FAKE."""

    def __init__(
        self,
        root: Path,
        image_size: int = 224,
        training: bool = True,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.training = training

        self.samples: List[Tuple[Path, int]] = []
        for label, cls in enumerate(["REAL", "FAKE"]):
            folder = self.root / cls
            if not folder.exists():
                raise FileNotFoundError(f"Expected folder: {folder}")
            for p in folder.rglob("*"):
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                    self.samples.append((p, label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under {self.root}. Ensure folders REAL/ and FAKE/ contain images.")

        self.rgb_tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

        self.aug_tf = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02
                        )
                    ],
                    p=0.5,
                ),
            ]
        )

        self.rgb_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.ela_norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.training:
            img = self.aug_tf(img)

        ela = compute_ela_rgb(img)

        rgb_t = self.rgb_tf(img)
        ela_t = self.rgb_tf(ela)
        rgb_t = self.rgb_norm(rgb_t)
        ela_t = self.ela_norm(ela_t)

        x = torch.cat([rgb_t, ela_t], dim=0)
        y = torch.tensor([float(label)], dtype=torch.float32)
        return x, y, str(path)


def split_samples(samples: List[Tuple[Path, int]], val_ratio: float, seed: int):
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)

    val_n = int(round(len(shuffled) * val_ratio))
    val = shuffled[:val_n]
    train = shuffled[val_n:]
    return train, val


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    bce = nn.BCEWithLogitsLoss()

    for x, y, _paths in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = bce(logits, y)
        total_loss += float(loss.item()) * x.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        correct += int((preds == y).sum().item())
        total += x.size(0)

    return total_loss / max(1, total), correct / max(1, total)


def train(
    data_dir: Path,
    out_dir: Path,
    base_model: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    image_size: int,
    val_ratio: float,
    seed: int,
) -> Path:
    seed_everything(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = RealFakeDataset(data_dir, image_size=image_size, training=True)
    samples = dataset.samples
    train_samples, val_samples = split_samples(samples, val_ratio=val_ratio, seed=seed)

    train_ds = RealFakeDataset(data_dir, image_size=image_size, training=True)
    val_ds = RealFakeDataset(data_dir, image_size=image_size, training=False)

    # Ensure we only use the splits
    train_ds.samples = train_samples
    val_ds.samples = val_samples

    num_workers = min(8, max(0, (os.cpu_count() or 4) - 1))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet6Ch(base=base_model, pretrained=True).to(device)

    y_train = np.array([label for _path, label in train_samples], dtype=np.int64)
    pos = int(y_train.sum())
    neg = int((y_train == 0).sum())
    if pos > 0 and neg > 0:
        pos_weight = torch.tensor([neg / max(1, pos)], dtype=torch.float32, device=device)
    else:
        pos_weight = torch.tensor([1.0], dtype=torch.float32, device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, math.ceil(len(train_loader)))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=lr,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=0.1,
    )

    best_val_acc = -1.0
    best_path = out_dir / "best_model.pt"

    print(f"[info] Device: {device}")
    print(f"[info] Train/Val: {len(train_samples)} / {len(val_samples)}")
    print(f"[info] pos_weight={float(pos_weight.item()):.3f} (neg={neg}, pos={pos})")
    print(f"[info] Starting training...\n")

    training_start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        model.train()
        running = 0.0
        seen = 0

        for x, y, _paths in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = bce(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            running += float(loss.item()) * x.size(0)
            seen += x.size(0)

        train_loss = running / max(1, seen)
        val_loss, val_acc = evaluate(model, val_loader, device=device)
        
        epoch_time = time.time() - epoch_start_time
        elapsed_time = time.time() - training_start_time
        avg_time_per_epoch = elapsed_time / epoch
        remaining_time = avg_time_per_epoch * (epochs - epoch)

        print(f"[epoch {epoch:02d}/{epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} | " 
              f"time: {epoch_time:.2f}s | elapsed: {elapsed_time/60:.1f}m | ETA: {remaining_time/60:.1f}m")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "base_model": base_model,
                    "image_size": image_size,
                },
                best_path,
            )

    total_training_time = time.time() - training_start_time
    print(f"\n[done] Best val_acc={best_val_acc:.4f}, saved {best_path}")
    print(f"[info] Total training time: {total_training_time/60:.2f} minutes ({total_training_time/3600:.2f} hours)")
    return best_path


def main():
    parser = argparse.ArgumentParser(description="Train REAL-vs-FAKE detector")
    parser.add_argument("--data-dir", type=str, default="train", help="Path to train/ folder")
    parser.add_argument("--out-dir", type=str, default="runs/real_vs_ai", help="Where to save checkpoints")
    parser.add_argument("--base-model", type=str, default="resnet18", choices=["resnet18", "resnet50"], help="Backbone model")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction of data used for validation")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    train(
        data_dir=Path(args.data_dir),
        out_dir=Path(args.out_dir),
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
