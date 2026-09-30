"""Minimal paired-image trainer for HazeWaveNet.

Expected layout: DATA/hazy/* and DATA/clear/*. Hazy suffixes such as
``_hazy`` are removed automatically when finding the clear counterpart.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import random
from tqdm.auto import tqdm
from .model import HazeWaveNet, haze_wavelet_loss

class PairedImages(Dataset):
    def __init__(self, root: Path, size: int = 256, augment: bool = False):
        self.hazy, self.clear, self.size, self.augment = root / "hazy", root / "clear", size, augment
        if not self.hazy.is_dir() or not self.clear.is_dir():
            raise ValueError(f"Expected directories {self.hazy} and {self.clear}")

        clear_by_stem = {p.stem.lower(): p for p in self.clear.iterdir() if p.is_file()}
        self.items = []
        for hazy_path in sorted(p for p in self.hazy.iterdir() if p.is_file()):
            stem = hazy_path.stem.lower()
            base = stem.rsplit("_", 1)[0] if stem.rsplit("_", 1)[-1].isdigit() else stem
            candidates = (stem, stem.removesuffix("_hazy"), stem.removesuffix("-hazy"), base, stem.split("_", 1)[0])
            clear_path = next((clear_by_stem[s] for s in candidates if s in clear_by_stem), None)
            if clear_path is not None:
                self.items.append((hazy_path, clear_path))
        if not self.items:
            raise ValueError(f"No matching images in {self.hazy} and {self.clear}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        out = []
        for path in self.items[i]:
            image = Image.open(path).convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR)
            out.append(torch.from_numpy(np.array(image, copy=True)).float().div(255).permute(2, 0, 1))
        if self.augment:
            if random.random() < 0.5:
                out = [torch.flip(x, (2,)) for x in out]
            if random.random() < 0.5:
                out = [torch.flip(x, (1,)) for x in out]
            k = random.randint(0, 3)
            out = [torch.rot90(x, k, (1, 2)) for x in out]
            brightness, contrast = random.uniform(-0.1, 0.1), random.uniform(0.9, 1.1)
            out[0] = ((out[0] - 0.5) * contrast + 0.5 + brightness).clamp(0, 1)
        return tuple(out)


def image_metrics(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    """Return batch-averaged PSNR and global SSIM (images are in [0, 1])."""
    pred = pred.clamp(0.0, 1.0)
    mse = (pred - target).square().flatten(1).mean(1).clamp_min(1e-10)
    psnr = (10.0 * torch.log10(1.0 / mse)).mean()
    mu_x, mu_y = pred.mean((2, 3), keepdim=True), target.mean((2, 3), keepdim=True)
    var_x = (pred - mu_x).square().mean((2, 3), keepdim=True)
    var_y = (target - mu_y).square().mean((2, 3), keepdim=True)
    cov = ((pred - mu_x) * (target - mu_y)).mean((2, 3), keepdim=True)
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov + c2) /
            ((mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2))).mean()
    return psnr.item(), ssim.item()


@torch.no_grad()
def evaluate(model: HazeWaveNet, loader: DataLoader, device: str) -> tuple[float, float, float]:
    model.eval(); total_loss = total_psnr = total_ssim = 0.0; batches = 0
    for hazy, clear in loader:
        hazy, clear = hazy.to(device), clear.to(device)
        pred = model(hazy)
        total_loss += haze_wavelet_loss(pred, clear).item()
        psnr, ssim = image_metrics(pred, clear)
        total_psnr += psnr; total_ssim += ssim; batches += 1
    return total_loss / batches, total_psnr / batches, total_ssim / batches

def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("data", type=Path); ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4); ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--scheduler", choices=("none", "cosine", "step"), default="none")
    ap.add_argument("--min-lr", type=float, default=1e-6); ap.add_argument("--step-size", type=int, default=100)
    ap.add_argument("--augment", action="store_true"); ap.add_argument("--multi-scale", action="store_true")
    ap.add_argument("--patience", type=int, default=0, help="Early stopping patience; 0 disables it")
    ap.add_argument("--out", type=Path, default=Path("haze_wavelet.pt"))
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--multi-gpu", action="store_true", help="Use all visible CUDA devices via DataParallel")
    ap.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes")
    ap.add_argument("--val-data", type=Path, default=None,
                    help="Validation directory containing hazy/ and clear/ (default: sibling val directory)")
    ap.add_argument("--history", type=Path, default=None,
                    help="Optional JSON file for per-epoch train/validation metrics")
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"device={device}")
    loader = DataLoader(PairedImages(args.data, args.size, args.augment), args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device == "cuda", persistent_workers=args.num_workers > 0)
    val_root = args.val_data or args.data.parent / "val"
    val_loader = DataLoader(PairedImages(val_root, args.size), args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device == "cuda", persistent_workers=args.num_workers > 0) if val_root.is_dir() else None
    print(f"train_images={len(loader.dataset)} train_batches={len(loader)} batch_size={args.batch_size} workers={args.num_workers}", flush=True)
    model = HazeWaveNet().to(device)
    if args.multi_gpu:
        if device != "cuda" or not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError("--multi-gpu requires at least two visible CUDA devices")
        model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.min_lr)
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)
    history = []
    best_val_loss, stale_epochs = float("inf"), 0
    show_tqdm = os.environ.get("HW_TQDM", "auto") == "1" or (os.environ.get("HW_TQDM", "auto") == "auto" and sys.stdout.isatty())
    epoch_bar = tqdm(range(args.epochs), desc="Training", unit="epoch", disable=not show_tqdm)
    show_batch_progress = show_tqdm and os.environ.get("HW_TQDM_BATCH", "0") == "1"
    for epoch in epoch_bar:
        model.train(); total = total_psnr = total_ssim = 0.0; batches = 0
        # Nested tqdm redraws become one line per update in Colab's `!python` output.
        batch_bar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}", unit="batch", leave=False, disable=not show_batch_progress)
        for batch_index, (hazy, clear) in enumerate(batch_bar, start=1):
            hazy, clear = hazy.to(device), clear.to(device); optimizer.zero_grad(set_to_none=True)
            if args.multi_scale:
                scale = random.choice((64, 128, 256))
                hazy = torch.nn.functional.interpolate(hazy, (scale, scale), mode="bilinear", align_corners=False)
                clear = torch.nn.functional.interpolate(clear, (scale, scale), mode="bilinear", align_corners=False)
            pred = model(hazy); loss = haze_wavelet_loss(pred, clear); loss.backward(); optimizer.step()
            psnr, ssim = image_metrics(pred.detach(), clear)
            total += loss.item(); total_psnr += psnr; total_ssim += ssim; batches += 1
            batch_bar.set_postfix(loss=f"{total / batches:.5f}", psnr=f"{total_psnr / batches:.2f}", ssim=f"{total_ssim / batches:.4f}")
            if not show_tqdm and (batch_index == 1 or batch_index == len(loader) or batch_index % max(1, len(loader) // 10) == 0):
                print(f"epoch {epoch + 1:03d}/{args.epochs} batch {batch_index}/{len(loader)} | loss={total / batches:.5f} PSNR={total_psnr / batches:.2f} SSIM={total_ssim / batches:.4f}", flush=True)
        train_values = (total / batches, total_psnr / batches, total_ssim / batches)
        if val_loader is not None:
            val_values = evaluate(model, val_loader, device)
            history.append({"epoch": epoch + 1, "train_loss": train_values[0], "train_psnr": train_values[1], "train_ssim": train_values[2], "val_loss": val_values[0], "val_psnr": val_values[1], "val_ssim": val_values[2]})
            epoch_bar.set_postfix(loss=f"{train_values[0]:.5f}", psnr=f"{train_values[1]:.2f}", val_loss=f"{val_values[0]:.5f}", val_psnr=f"{val_values[1]:.2f}")
            epoch_bar.write(f"epoch {epoch + 1:03d}/{args.epochs} | "
                            f"train loss={train_values[0]:.5f} PSNR={train_values[1]:.2f} SSIM={train_values[2]:.4f} | "
                            f"val loss={val_values[0]:.5f} PSNR={val_values[1]:.2f} SSIM={val_values[2]:.4f}")
            if val_values[0] < best_val_loss:
                best_val_loss, stale_epochs = val_values[0], 0
                state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
                torch.save({"model": state, "epoch": epoch + 1}, args.out)
            else:
                stale_epochs += 1
        else:
            history.append({"epoch": epoch + 1, "train_loss": train_values[0], "train_psnr": train_values[1], "train_ssim": train_values[2]})
            epoch_bar.set_postfix(loss=f"{train_values[0]:.5f}", psnr=f"{train_values[1]:.2f}", ssim=f"{train_values[2]:.4f}")
            epoch_bar.write(f"epoch {epoch + 1:03d}/{args.epochs} | "
                            f"train loss={train_values[0]:.5f} PSNR={train_values[1]:.2f} SSIM={train_values[2]:.4f} | val unavailable")
        if scheduler is not None:
            scheduler.step()
        if args.patience and stale_epochs >= args.patience:
            epoch_bar.write(f"Early stopping at epoch {epoch + 1}; validation loss did not improve for {args.patience} epochs.")
            break
    if val_loader is None:
        state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
        torch.save({"model": state, "epoch": len(history)}, args.out)
    if args.history:
        args.history.parent.mkdir(parents=True, exist_ok=True)
        args.history.write_text(json.dumps(history, indent=2), encoding="utf-8")

if __name__ == "__main__": main()
