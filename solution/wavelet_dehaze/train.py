"""Minimal paired-image trainer for HazeWaveNet.

Expected layout: DATA/hazy/* and DATA/clear/*. Hazy suffixes such as
``_hazy`` are removed automatically when finding the clear counterpart.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from .model import HazeWaveNet, haze_wavelet_loss

class PairedImages(Dataset):
    def __init__(self, root: Path, size: int = 256):
        self.hazy, self.clear, self.size = root / "hazy", root / "clear", size
        if not self.hazy.is_dir() or not self.clear.is_dir():
            raise ValueError(f"Expected directories {self.hazy} and {self.clear}")

        clear_by_stem = {p.stem.lower(): p for p in self.clear.iterdir() if p.is_file()}
        self.items = []
        for hazy_path in sorted(p for p in self.hazy.iterdir() if p.is_file()):
            stem = hazy_path.stem.lower()
            candidates = (stem, stem.removesuffix("_hazy"), stem.removesuffix("-hazy"))
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
    ap.add_argument("--out", type=Path, default=Path("haze_wavelet.pt"))
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--val-data", type=Path, default=None,
                    help="Validation directory containing hazy/ and clear/ (default: sibling val directory)")
    ap.add_argument("--history", type=Path, default=None,
                    help="Optional JSON file for per-epoch train/validation metrics")
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"device={device}")
    loader = DataLoader(PairedImages(args.data, args.size), args.batch_size, shuffle=True, num_workers=0)
    val_root = args.val_data or args.data.parent / "val"
    val_loader = DataLoader(PairedImages(val_root, args.size), args.batch_size, shuffle=False, num_workers=0) if val_root.is_dir() else None
    model = HazeWaveNet().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    history = []
    for epoch in range(args.epochs):
        model.train(); total = total_psnr = total_ssim = 0.0; batches = 0
        for hazy, clear in loader:
            hazy, clear = hazy.to(device), clear.to(device); optimizer.zero_grad(set_to_none=True)
            pred = model(hazy); loss = haze_wavelet_loss(pred, clear); loss.backward(); optimizer.step()
            psnr, ssim = image_metrics(pred.detach(), clear)
            total += loss.item(); total_psnr += psnr; total_ssim += ssim; batches += 1
        train_values = (total / batches, total_psnr / batches, total_ssim / batches)
        if val_loader is not None:
            val_values = evaluate(model, val_loader, device)
            history.append({"epoch": epoch + 1, "train_loss": train_values[0], "train_psnr": train_values[1], "train_ssim": train_values[2], "val_loss": val_values[0], "val_psnr": val_values[1], "val_ssim": val_values[2]})
            print(f"epoch {epoch + 1:03d}/{args.epochs} | "
                  f"train loss={train_values[0]:.5f} PSNR={train_values[1]:.2f} SSIM={train_values[2]:.4f} | "
                  f"val loss={val_values[0]:.5f} PSNR={val_values[1]:.2f} SSIM={val_values[2]:.4f}")
        else:
            history.append({"epoch": epoch + 1, "train_loss": train_values[0], "train_psnr": train_values[1], "train_ssim": train_values[2]})
            print(f"epoch {epoch + 1:03d}/{args.epochs} | "
                  f"train loss={train_values[0]:.5f} PSNR={train_values[1]:.2f} SSIM={train_values[2]:.4f} | val unavailable")
    torch.save({"model": model.state_dict()}, args.out)
    if args.history:
        args.history.parent.mkdir(parents=True, exist_ok=True)
        args.history.write_text(json.dumps(history, indent=2), encoding="utf-8")

if __name__ == "__main__": main()
