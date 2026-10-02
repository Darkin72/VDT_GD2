from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_training(label, data, output, args, spec):
    slug = label.lower().replace("-", "_")
    checkpoint = output / f"haze_wavelet_{slug}.pt"
    history = output / f"history_{slug}.json"
    effective_batch = args.effective_batch_size or spec["batch"]
    if effective_batch < 1:
        raise ValueError("--effective-batch-size must be positive")
    physical_batch = min(args.micro_batch_size, effective_batch)
    command = [sys.executable, "-m", "solution.wavelet_dehaze.train", str(data),
               "--device", args.device, "--epochs", str(spec["epochs"]),
               "--batch-size", str(effective_batch), "--micro-batch-size", str(physical_batch),
               "--size", str(args.size), "--lr", str(spec["lr"]), "--weight-decay", str(args.weight_decay),
               "--scheduler", spec["scheduler"], "--min-lr", str(args.min_lr),
               "--step-size", str(args.step_size), "--patience", str(args.patience),
               "--num-workers", str(args.num_workers), "--prefetch-factor", str(args.prefetch_factor),
               "--out", str(checkpoint), "--history", str(history), "--log-every", "1", "--metrics-every", "10",
               "--cache-dir", str(args.output_dir / "cache" / slug), "--amp-dtype", args.amp_dtype]
    if spec.get("val_data") and spec["val_data"].is_dir():
        command += ["--val-data", str(spec["val_data"])]
    else:
        command += ["--val-fraction", str(args.val_fraction)]
    for flag, enabled in (("--augment", args.augment), ("--multi-scale", args.multi_scale),
                          ("--pin-memory", args.pin_memory), ("--persistent-workers", args.persistent_workers),
                          ("--amp", args.amp)):
        if enabled:
            command.append(flag)
    print(f"[TRAIN {label}] epochs={spec['epochs']} batch={effective_batch} micro={physical_batch} lr={spec['lr']} scheduler={spec['scheduler']}", flush=True)
    subprocess.run(command, check=True)
    return checkpoint, history


def plot_histories(histories, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = (("loss", "train_loss", "val_loss", "Loss"), ("psnr", "train_psnr", "val_psnr", "PSNR (dB)"), ("ssim", "train_ssim", "val_ssim", "SSIM"))
    for label, history_path in histories.items():
        records = json.loads(history_path.read_text(encoding="utf-8"))
        epochs = [row["epoch"] for row in records]
        for slug, train_key, val_key, title in metrics:
            fig, axis = plt.subplots(figsize=(9, 5))
            axis.plot(epochs, [row[train_key] for row in records], label="train")
            if records and val_key in records[0]:
                axis.plot(epochs, [row[val_key] for row in records], label="validation")
            axis.set(xlabel="Epoch", ylabel=title, title=f"{label} - {title}")
            axis.grid(True, alpha=0.25); axis.legend(); fig.tight_layout()
            fig.savefig(output / f"{label.lower().replace('-', '_')}_{slug}.png", dpi=160)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--i-haze-data", type=Path, required=True)
    parser.add_argument("--o-hazy-data", type=Path, required=True)
    parser.add_argument("--its-data", type=Path, required=True)
    parser.add_argument("--ots-data", type=Path, required=True)
    parser.add_argument("--eval-data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/hazewavenet"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--micro-batch-size", type=int, default=16)
    parser.add_argument("--effective-batch-size", type=int, default=None,
                        help="Override notebook effective batch for all datasets; micro batch remains physical per forward")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--step-size", type=int, default=100)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--profile-repeats", type=int, default=10)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--multi-scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", choices=("float16", "bfloat16"), default="bfloat16")
    args = parser.parse_args()
    if args.device == "auto":
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = {
        "I-HAZE": {"data": args.i_haze_data, "epochs": 500, "batch": 8, "lr": 2e-4, "scheduler": "step", "val_data": args.i_haze_data.parent / "val"},
        "O-HAZY": {"data": args.o_hazy_data, "epochs": 500, "batch": 8, "lr": 2e-4, "scheduler": "step", "val_data": args.o_hazy_data.parent / "val"},
        "SOTS-ITS": {"data": args.its_data, "epochs": 1000, "batch": 16, "lr": 2e-4, "scheduler": "cosine"},
        "SOTS-OTS": {"data": args.ots_data, "epochs": 1000, "batch": 16, "lr": 1e-4, "scheduler": "cosine"},
    }
    checkpoints, histories = {}, {}
    print("[1/3] Training four datasets", flush=True)
    for label, spec in specs.items():
        checkpoints[label], histories[label] = run_training(label, spec["data"], args.output_dir, args, spec)
    print("[2/3] Writing metric plots", flush=True)
    plot_histories(histories, args.output_dir)
    print("[3/3] Evaluating with detailed profiling", flush=True)
    command = [sys.executable, "-m", "solution.wavelet_dehaze.evaluate", "--data-root", str(args.eval_data_root),
               "--output-dir", str(args.output_dir / "reports"), "--device", args.device,
               "--max-side", "1024", "--profile-repeats", str(args.profile_repeats),
               "--save-images-dir", str(args.output_dir / "inference")]
    for option, label in (("--checkpoint-i-haze", "I-HAZE"), ("--checkpoint-o-hazy", "O-HAZY"),
                          ("--checkpoint-sots-its", "SOTS-ITS"), ("--checkpoint-sots-ots", "SOTS-OTS")):
        command += [option, str(checkpoints[label])]
    subprocess.run(command, check=True)
    print(f"Completed. Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()