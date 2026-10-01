"""Train ITS/OTS, save six plots and evaluate both checkpoints on four test sets."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def paired_root(root: Path) -> Path:
    if (root / "hazy").is_dir() and (root / "clear").is_dir():
        return root
    candidates = sorted({folder.parent for folder in root.rglob("hazy")
                         if folder.is_dir() and (folder.parent / "clear").is_dir()})
    if len(candidates) != 1:
        raise ValueError(f"Expected one paired hazy/clear directory under {root}, found {candidates}")
    return candidates[0]


def run_training(name: str, data: Path, output: Path, args: argparse.Namespace) -> Path:
    checkpoint = output / f"haze_wavelet_{name.lower()}.pt"
    history = output / f"history_{name.lower()}.json"
    command = [
        sys.executable,
        "-m",
        "solution.wavelet_dehaze.train",
        str(data),
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--micro-batch-size",
        str(args.micro_batch_size),
        "--size",
        str(args.size),
        "--lr",
        str(args.its_lr if name == "ITS" else args.ots_lr),
        "--weight-decay",
        str(args.weight_decay),
        "--scheduler",
        args.scheduler,
        "--min-lr",
        str(args.min_lr),
        "--patience",
        str(args.patience),
        "--val-fraction",
        str(args.val_fraction),
        "--num-workers",
        str(args.num_workers),
        "--prefetch-factor",
        str(args.prefetch_factor),
        "--out",
        str(checkpoint),
        "--history",
        str(history),
    ]
    if args.augment:
        command.append("--augment")
    if args.multi_scale:
        command.append("--multi-scale")
    if args.pin_memory:
        command.append("--pin-memory")
    if args.persistent_workers:
        command.append("--persistent-workers")
    if args.amp:
        command.append("--amp")
    subprocess.run(command, check=True)
    return history


def plot_histories(histories: dict[str, Path], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = (
        ("loss", "train_loss", "val_loss", "Loss"),
        ("psnr", "train_psnr", "val_psnr", "PSNR (dB)"),
        ("ssim", "train_ssim", "val_ssim", "SSIM"),
    )
    for name, history_path in histories.items():
        records = json.loads(history_path.read_text(encoding="utf-8"))
        if not records:
            raise ValueError(f"Empty training history: {history_path}")
        epochs = [record["epoch"] for record in records]
        for slug, train_key, val_key, label in metrics:
            figure, axis = plt.subplots(figsize=(9, 5))
            axis.plot(epochs, [record[train_key] for record in records], label="train")
            if val_key in records[0]:
                axis.plot(epochs, [record[val_key] for record in records], label="validation")
            axis.set_xlabel("Epoch")
            axis.set_ylabel(label)
            axis.set_title(f"{name} - {label}")
            axis.grid(True, alpha=0.25)
            axis.legend()
            figure.tight_layout()
            figure.savefig(output / f"{name.lower()}_{slug}.png", dpi=160)
            plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--its-data", type=Path, required=True)
    parser.add_argument("--ots-data", type=Path, required=True)
    parser.add_argument("--eval-data-root", type=Path, required=True)
    parser.add_argument("--eval-max-side", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/hazewavenet"))
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--its-lr", type=float, default=2e-4)
    parser.add_argument("--ots-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=("none", "cosine", "step"), default="cosine")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--multi-scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    its_data, ots_data = paired_root(args.its_data), paired_root(args.ots_data)
    from .evaluate import pairs
    test_roots = {
        "i-haze": args.eval_data_root / "I-HAZE",
        "o-hazy": args.eval_data_root / "O-HAZY",
        "sots-indoor": args.eval_data_root / "Synthetic Objective Testing Set (SOTS) [RESIDE]",
        "sots-outdoor": args.eval_data_root / "Synthetic Objective Testing Set (SOTS) [RESIDE]",
    }
    for dataset, root in test_roots.items():
        if not pairs(root, dataset):
            raise ValueError(f"No matching test images for {dataset} under {root}")
    if args.device == "auto":
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("ITS", its_data), ("OTS", ots_data)):
        history = run_training(name, data, args.output_dir, args)
        plot_histories({name: history}, args.output_dir)
    subprocess.run([
        sys.executable, "-m", "solution.wavelet_dehaze.evaluate",
        "--data-root", str(args.eval_data_root),
        "--output-dir", str(args.output_dir / "reports"),
        "--checkpoint-i-haze", str(args.output_dir / "haze_wavelet_its.pt"),
        "--checkpoint-o-hazy", str(args.output_dir / "haze_wavelet_ots.pt"),
        "--checkpoint-sots-its", str(args.output_dir / "haze_wavelet_its.pt"),
        "--checkpoint-sots-ots", str(args.output_dir / "haze_wavelet_ots.pt"),
        "--device", args.device, "--max-side", str(args.eval_max_side),
    ], check=True)
    print("Two checkpoints, six plots and four XLSX reports saved to", args.output_dir)


if __name__ == "__main__":
    main()
