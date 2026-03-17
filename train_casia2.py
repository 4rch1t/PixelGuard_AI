import argparse
import io
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageChops, ImageEnhance
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_list_file(path: Path) -> List[str]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


def safe_open_image(path: Path) -> Image.Image:
    # CASIA2 includes .jpg and .tif; PIL can open both.
    img = Image.open(path)
    # Normalize to RGB; we don't want alpha channels.
    return img.convert("RGB")


def compute_ela_rgb(img_rgb: Image.Image, quality: int = 90, enhance: int = 15) -> Image.Image:
    """
    Error Level Analysis:
    - Recompress image to JPEG at a fixed quality
    - Take pixel-wise difference
    - Enhance for visibility
    Returns RGB image in [0,255].
    """
    buf = io.BytesIO()
    # Convert to RGB just in case.
    img_rgb.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    jpeg = Image.open(buf).convert("RGB")
    diff = ImageChops.difference(img_rgb, jpeg)
    diff = ImageEnhance.Brightness(diff).enhance(enhance)
    return diff


@dataclass
class Sample:
    path: Path
    label: int  # 0 authentic, 1 tampered


class Casia2Dataset(Dataset):
    def __init__(
        self,
        samples: List[Sample],
        image_size: int = 224,
        ela_quality: int = 90,
        ela_enhance: int = 15,
        training: bool = True,
    ) -> None:
        self.samples = samples
        self.image_size = image_size
        self.ela_quality = ela_quality
        self.ela_enhance = ela_enhance
        self.training = training

        self.rgb_tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

        # Simple, robust augmentations (apply to RGB only; ELA is derived after).
        self.aug_tf = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02)],
                    p=0.5,
                ),
            ]
        )

        # ImageNet normalization for RGB; keep ELA unnormalized then scale similarly.
        self.rgb_norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.ela_norm = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = safe_open_image(s.path)
        if self.training:
            img = self.aug_tf(img)

        ela = compute_ela_rgb(img, quality=self.ela_quality, enhance=self.ela_enhance)

        rgb_t = self.rgb_tf(img)
        ela_t = self.rgb_tf(ela)

        rgb_t = self.rgb_norm(rgb_t)
        ela_t = self.ela_norm(ela_t)

        x = torch.cat([rgb_t, ela_t], dim=0)  # (6,H,W)
        y = torch.tensor([float(s.label)], dtype=torch.float32)
        return x, y, str(s.path)


def build_samples(dataset_root: Path) -> List[Sample]:
    au_list = dataset_root / "au_list.txt"
    tp_list = dataset_root / "tp_list.txt"
    au_dir = dataset_root / "Au"
    tp_dir = dataset_root / "Tp"

    if not au_list.exists() or not tp_list.exists():
        raise FileNotFoundError(
            f"Expected list files at {au_list} and {tp_list}. (Your dataset_root should be CASIA2.0_revised/)"
        )
    if not au_dir.exists() or not tp_dir.exists():
        raise FileNotFoundError(f"Expected image folders {au_dir} and {tp_dir}.")

    au_files = read_list_file(au_list)
    tp_files = read_list_file(tp_list)

    samples: List[Sample] = []
    missing = 0

    for fn in au_files:
        p = au_dir / fn
        if p.exists():
            samples.append(Sample(path=p, label=0))
        else:
            missing += 1

    for fn in tp_files:
        p = tp_dir / fn
        if p.exists():
            samples.append(Sample(path=p, label=1))
        else:
            missing += 1

    if len(samples) == 0:
        raise RuntimeError("No images found. Check that CASIA2 images exist under Au/ and Tp/.")
    if missing > 0:
        print(f"[warn] Missing {missing} files referenced by list(s). Continuing with {len(samples)} found samples.")

    return samples


def split_samples(samples: List[Sample], val_ratio: float, seed: int) -> Tuple[List[Sample], List[Sample]]:
    rng = random.Random(seed)
    samples = samples[:]
    rng.shuffle(samples)

    val_n = int(round(len(samples) * val_ratio))
    val = samples[:val_n]
    train = samples[val_n:]
    return train, val


class ResNet6Ch(nn.Module):
    def __init__(self, base: str = "resnet18", pretrained: bool = True) -> None:
        super().__init__()
        if base == "resnet18":
            m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            feat_dim = 512
        elif base == "resnet50":
            m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            feat_dim = 2048
        else:
            raise ValueError("base must be resnet18 or resnet50")

        # Replace first conv to accept 6 channels (RGB + ELA RGB).
        old = m.conv1
        new = nn.Conv2d(6, old.out_channels, kernel_size=old.kernel_size, stride=old.stride, padding=old.padding, bias=False)
        with torch.no_grad():
            # Initialize: copy ImageNet weights for first 3 channels; duplicate for ELA channels.
            new.weight[:, :3, :, :] = old.weight
            new.weight[:, 3:, :, :] = old.weight
        m.conv1 = new

        # Replace classifier head for binary logit.
        m.fc = nn.Linear(feat_dim, 1)
        self.backbone = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # logits shape (B,1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
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
    dataset_root: Path,
    out_dir: Path,
    base_model: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    image_size: int,
    val_ratio: float,
    seed: int,
    ela_quality: int,
    ela_enhance: int,
) -> Path:
    seed_everything(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = build_samples(dataset_root)
    train_s, val_s = split_samples(samples, val_ratio=val_ratio, seed=seed)

    train_ds = Casia2Dataset(
        train_s,
        image_size=image_size,
        ela_quality=ela_quality,
        ela_enhance=ela_enhance,
        training=True,
    )
    val_ds = Casia2Dataset(
        val_s,
        image_size=image_size,
        ela_quality=ela_quality,
        ela_enhance=ela_enhance,
        training=False,
    )

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

    # Handle class imbalance with pos_weight.
    y_train = np.array([s.label for s in train_s], dtype=np.int64)
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
    print(f"[info] Train/Val: {len(train_s)} / {len(val_s)}")
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
                    "ela_quality": ela_quality,
                    "ela_enhance": ela_enhance,
                },
                best_path,
            )

    total_training_time = time.time() - training_start_time
    print(f"\n[done] Best val_acc={best_val_acc:.4f}, saved {best_path}")
    print(f"[info] Total training time: {total_training_time/60:.2f} minutes ({total_training_time/3600:.2f} hours)")
    return best_path


def ascii_confidence_bar(p: float, width: int = 30) -> str:
    p = float(max(0.0, min(1.0, p)))
    filled = int(round(p * width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + f"] {p*100:.1f}%"


def load_model(ckpt_path: Path, device: torch.device) -> Tuple[nn.Module, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = ResNet6Ch(base=ckpt.get("base_model", "resnet18"), pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def tensor_from_path(
    img_path: Path,
    image_size: int,
    ela_quality: int,
    ela_enhance: int,
) -> Tuple[torch.Tensor, float]:
    img = safe_open_image(img_path)
    ela = compute_ela_rgb(img, quality=ela_quality, enhance=ela_enhance)

    tf = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    rgb_t = tf(img)
    ela_t = tf(ela)

    rgb_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ela_norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    x = torch.cat([rgb_norm(rgb_t), ela_norm(ela_t)], dim=0).unsqueeze(0)

    # Simple artifact/anomaly score: mean ELA intensity (0..1-ish after ToTensor).
    ela_mean = float(ela_t.mean().item())
    return x, ela_mean


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._acts = None
        self._grads = None
        self._hooks = []

        def fwd_hook(_m, _inp, out):
            self._acts = out

        def bwd_hook(_m, _gin, gout):
            self._grads = gout[0]

        self._hooks.append(self.target_layer.register_forward_hook(fwd_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(bwd_hook))

    def close(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        score = logits.squeeze()
        score.backward(retain_graph=False)

        acts = self._acts  # (B,C,H,W)
        grads = self._grads  # (B,C,H,W)
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (B,C,1,1)
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (B,1,H,W)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-6)
        return cam  # (B,1,H,W) in [0,1]


def save_heatmap_overlay(img_path: Path, cam_1hw: torch.Tensor, out_path: Path, alpha: float = 0.45) -> None:
    img = safe_open_image(img_path)
    w, h = img.size

    cam = cam_1hw.squeeze().detach().cpu().numpy()  # (H,W)
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize((w, h), resample=Image.BILINEAR)
    cam_img = cam_img.convert("L")

    # Simple "jet-like" colormap without extra deps.
    cam_np = np.array(cam_img, dtype=np.float32) / 255.0
    r = np.clip(1.5 * cam_np, 0, 1)
    g = np.clip(1.5 * (1 - np.abs(cam_np - 0.5) * 2), 0, 1)
    b = np.clip(1.5 * (1 - cam_np), 0, 1)
    heat = np.stack([r, g, b], axis=-1)
    heat = Image.fromarray(np.uint8(heat * 255)).convert("RGBA")

    base = img.convert("RGBA")
    heat.putalpha(int(255 * alpha))
    overlay = Image.alpha_composite(base, heat)
    overlay.save(out_path)


@torch.no_grad()
def predict(
    ckpt_path: Path,
    image_path: Path,
    heatmap_out: Path | None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_model(ckpt_path, device=device)

    x, ela_mean = tensor_from_path(
        image_path,
        image_size=int(meta.get("image_size", 224)),
        ela_quality=int(meta.get("ela_quality", 90)),
        ela_enhance=int(meta.get("ela_enhance", 15)),
    )
    x = x.to(device)
    logits = model(x)
    p_tampered = float(torch.sigmoid(logits).item())

    label = "TAMPERED" if p_tampered >= 0.5 else "AUTHENTIC"
    conf = p_tampered if label == "TAMPERED" else (1.0 - p_tampered)

    print(f"Image: {image_path}")
    print(f"Prediction: {label}")
    print(f"Confidence: {ascii_confidence_bar(conf)}")
    print(f"Artifact score (ELA mean): {ela_mean:.4f} (higher often means more recompression/editing artifacts)")

    if heatmap_out is not None:
        # Use last conv block for CAM.
        # For torchvision resnet: layer4 is last stage.
        cam = GradCAM(model, model.backbone.layer4)
        x_req = x.clone().requires_grad_(True)
        cam_map = cam(x_req)
        cam.close()
        heatmap_out.parent.mkdir(parents=True, exist_ok=True)
        save_heatmap_overlay(image_path, cam_map, heatmap_out)
        print(f"Saved heatmap overlay: {heatmap_out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "predict"], required=True)

    ap.add_argument(
        "--dataset_root",
        type=str,
        default=str(Path("CASIA2.0_revised")),
        help="Path to CASIA2.0_revised (contains Au/, Tp/, au_list.txt, tp_list.txt).",
    )
    ap.add_argument("--out_dir", type=str, default="runs/casia2")
    ap.add_argument("--base_model", choices=["resnet18", "resnet50"], default="resnet18")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ela_quality", type=int, default=90)
    ap.add_argument("--ela_enhance", type=int, default=15)

    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--image", type=str, default="")
    ap.add_argument("--heatmap_out", type=str, default="")

    args = ap.parse_args()

    if args.mode == "train":
        best = train(
            dataset_root=Path(args.dataset_root),
            out_dir=Path(args.out_dir),
            base_model=args.base_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            image_size=args.image_size,
            val_ratio=args.val_ratio,
            seed=args.seed,
            ela_quality=args.ela_quality,
            ela_enhance=args.ela_enhance,
        )
        print(f"Best checkpoint: {best}")
        return

    if args.mode == "predict":
        if not args.ckpt or not args.image:
            raise SystemExit("--mode predict requires --ckpt and --image")
        heatmap_out = Path(args.heatmap_out) if args.heatmap_out else None
        predict(Path(args.ckpt), Path(args.image), heatmap_out)
        return


if __name__ == "__main__":
    main()

