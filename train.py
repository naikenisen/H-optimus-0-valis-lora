import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor

from dataset import VirtualStainingDataset
from model import HOptimusLoRA

def _gaussian_kernel_2d(size: int, sigma: float, channels: int) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-coords.pow(2) / (2 * sigma ** 2))
    kernel = g.unsqueeze(1) @ g.unsqueeze(0)
    kernel = kernel / kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1).contiguous()


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    channels = pred.shape[1]
    kernel = _gaussian_kernel_2d(window_size, sigma, channels).to(
        pred.device, pred.dtype
    )
    pad = window_size // 2

    mu1 = F.conv2d(pred, kernel, padding=pad, groups=channels)
    mu2 = F.conv2d(target, kernel, padding=pad, groups=channels)

    mu1_sq, mu2_sq, mu12 = mu1.pow(2), mu2.pow(2), mu1 * mu2

    sigma1_sq = F.conv2d(pred.pow(2), kernel, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target.pow(2), kernel, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, kernel, padding=pad, groups=channels) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


class CombinedLoss(nn.Module):

    def __init__(self, ssim_weight: float = 1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ssim_weight = ssim_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        mse_val = self.mse(pred, target)
        ssim_val = 1.0 - compute_ssim(pred, target)
        total = mse_val + self.ssim_weight * ssim_val
        return total, mse_val, ssim_val

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, optimizer, scaler, epoch, val_loss, config, path):
    torch.save(
        {
            "trainable_state_dict": model.get_trainable_state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "config": config,
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, scaler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_trainable_state_dict(ckpt["trainable_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("val_loss", float("inf"))


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    running_loss, running_mse, running_ssim = 0.0, 0.0, 0.0

    pbar = tqdm(loader, desc="  train", leave=False)
    for hes, cd30, _ in pbar:
        hes, cd30 = hes.to(device), cd30.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            pred = model(hes)
            loss, mse_l, ssim_l = criterion(pred, cd30)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * hes.size(0)
        running_mse += mse_l.item() * hes.size(0)
        running_ssim += ssim_l.item() * hes.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    n = len(loader.dataset)
    return running_loss / n, running_mse / n, running_ssim / n


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp):
    model.eval()
    running_loss, running_mse, running_ssim = 0.0, 0.0, 0.0

    for hes, cd30, _ in tqdm(loader, desc="  valid", leave=False):
        hes, cd30 = hes.to(device), cd30.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            pred = model(hes)
            loss, mse_l, ssim_l = criterion(pred, cd30)

        running_loss += loss.item() * hes.size(0)
        running_mse += mse_l.item() * hes.size(0)
        running_ssim += ssim_l.item() * hes.size(0)

    n = len(loader.dataset)
    return running_loss / n, running_mse / n, running_ssim / n


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune H-optimus-0 with LoRA")

    # Data
    p.add_argument("--data_dir", default="dataset", help="Root dataset directory")
    p.add_argument("--output_dir", default="checkpoints", help="Where to save models")

    # Model
    p.add_argument("--model_name", default="bioptimus/H-optimus-0")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--ssim_weight", type=float, default=1.0, help="Weight for SSIM loss")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true", default=True, help="Mixed precision")
    p.add_argument("--no_amp", dest="amp", action="store_false")

    # Resume
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")

    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    print(f"Device: {device} | Mixed precision: {use_amp}")

    # --- Image processor (normalization values) ---------------------------
    try:
        processor = AutoImageProcessor.from_pretrained(
            args.model_name, trust_remote_code=True
        )
        image_mean = tuple(processor.image_mean)
        image_std = tuple(processor.image_std)
        print(f"Processor normalisation — mean: {image_mean}, std: {image_std}")
    except Exception:
        image_mean = (0.485, 0.456, 0.406)
        image_std = (0.229, 0.224, 0.225)
        print("Using default ImageNet normalisation")

    # --- Datasets ---------------------------------------------------------
    train_ds = VirtualStainingDataset(
        root_dir=args.data_dir,
        split="train",
        image_size=args.image_size,
        mean=image_mean,
        std=image_std,
        augment=True,
    )
    valid_ds = VirtualStainingDataset(
        root_dir=args.data_dir,
        split="valid",
        image_size=args.image_size,
        mean=image_mean,
        std=image_std,
        augment=False,
    )
    print(f"Train: {len(train_ds)} pairs | Valid: {len(valid_ds)} pairs")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # --- Model ------------------------------------------------------------
    print("Building model...")
    model = HOptimusLoRA(
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_size=args.image_size,
    ).to(device)

    # --- Optimiser / Scheduler / Scaler -----------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    criterion = CombinedLoss(ssim_weight=args.ssim_weight).to(device)

    # --- Resume -----------------------------------------------------------
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(
            args.resume, model, optimizer, scaler, device
        )
        print(f"Resumed from epoch {start_epoch}, best val_loss {best_val_loss:.5f}")

    config = {
        "model_name": args.model_name,
        "image_size": args.image_size,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "ssim_weight": args.ssim_weight,
        "image_mean": image_mean,
        "image_std": image_std,
    }

    # --- Training loop ----------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        print(f"\nEpoch {epoch + 1}/{args.epochs}  (lr={scheduler.get_last_lr()[0]:.2e})")

        train_loss, train_mse, train_ssim = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )
        val_loss, val_mse, val_ssim = validate(
            model, valid_loader, criterion, device, use_amp
        )
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"  train — loss: {train_loss:.5f}  mse: {train_mse:.5f}  ssim: {train_ssim:.5f}\n"
            f"  valid — loss: {val_loss:.5f}  mse: {val_mse:.5f}  ssim: {val_ssim:.5f}  "
            f"({elapsed:.0f}s)"
        )

        # Save last checkpoint
        save_checkpoint(
            model, optimizer, scaler, epoch + 1, val_loss, config,
            os.path.join(args.output_dir, "last.pt"),
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, scaler, epoch + 1, val_loss, config,
                os.path.join(args.output_dir, "best.pt"),
            )
            print(f"  ★ New best model saved (val_loss={val_loss:.5f})")

    print(f"\nTraining finished. Best validation loss: {best_val_loss:.5f}")


if __name__ == "__main__":
    main()
