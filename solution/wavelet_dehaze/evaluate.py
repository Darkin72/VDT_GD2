"""Benchmark HazeWaveNet and export the four reports used by utils/evaluate.py."""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "utils"))
from excel_report import export_excel_reports
from evaluation_hardware import write_hardware_report
from .model import HazeWaveNet

def files(folder):
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"})

def pairs(root: Path, dataset: str):
    base = root / "test"
    if dataset == "sots-indoor": base = root / "indoor" / "test"
    if dataset == "sots-outdoor": base = root / "outdoor" / "test"
    clear = {p.stem.lower(): p for p in files(base / "clear")}
    result = []
    for hazy in files(base / "hazy"):
        key = hazy.stem.lower()
        if dataset == "i-haze": key = key.removesuffix("_hazy")
        if dataset.startswith("sots"): key = key.split("_")[0]
        if key in clear: result.append((hazy, clear[key]))
    return result

def run_model(model, pair_list, device, max_side):
    rows, elapsed_total = [], 0.0
    for hazy_path, clear_path in pair_list:
        hazy_img, clear_img = Image.open(hazy_path).convert("RGB"), Image.open(clear_path).convert("RGB")
        if max_side and max(hazy_img.size) > max_side:
            scale = max_side / max(hazy_img.size); size = (round(hazy_img.width * scale), round(hazy_img.height * scale))
            hazy_img, clear_img = hazy_img.resize(size, Image.Resampling.LANCZOS), clear_img.resize(size, Image.Resampling.LANCZOS)
        hazy = torch.from_numpy(np.array(hazy_img, copy=True)).float().div(255).permute(2, 0, 1).unsqueeze(0).to(device)
        clear = np.array(clear_img, copy=True)
        if device == "cuda": torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode(): output = model(hazy)
        if device == "cuda": torch.cuda.synchronize()
        elapsed = time.perf_counter() - start; elapsed_total += elapsed
        result = output[0].permute(1, 2, 0).mul(255).clamp(0, 255).byte().cpu().numpy()
        rows.append((peak_signal_noise_ratio(clear, result, data_range=255), structural_similarity(clear, result, channel_axis=2, data_range=255), elapsed * 1000))
    return rows, elapsed_total

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data-root", type=Path, required=True); ap.add_argument("--output-dir", type=Path, required=True)
    for name in ("i-haze", "o-hazy", "sots-its", "sots-ots"): ap.add_argument(f"--checkpoint-{name}", type=Path, default=None)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda"); ap.add_argument("--max-side", type=int, default=1024); args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA is not available")
    names = {"i-haze": "I-HAZE", "o-hazy": "O-HAZY", "sots-indoor": "Synthetic Objective Testing Set (SOTS) [RESIDE]", "sots-outdoor": "Synthetic Objective Testing Set (SOTS) [RESIDE]"}
    checkpoints = {key: getattr(args, "checkpoint_" + key.replace("-", "_")) for key in ("i-haze", "o-hazy", "sots-its", "sots-ots")}
    if not any(checkpoints.values()):
        raise ValueError("At least one checkpoint argument is required")
    datasets = {"i-haze": "i-haze", "o-hazy": "o-hazy", "sots-its": "sots-indoor", "sots-ots": "sots-outdoor"}
    all_rows, total_time, total_images = [], 0.0, 0
    for label, dataset in datasets.items():
        if checkpoints[label] is None:
            continue
        model = HazeWaveNet().to(args.device).eval(); state = torch.load(checkpoints[label], map_location=args.device); model.load_state_dict(state.get("model", state))
        root = args.data_root / ("I-HAZE" if dataset == "i-haze" else "O-HAZY" if dataset == "o-hazy" else names[dataset])
        pair_list = pairs(root, dataset)
        if not pair_list:
            raise ValueError(f"No matching test images for {dataset} under {root}")
        metrics, inference_time = run_model(model, pair_list, args.device, args.max_side)
        for (hazy, _), (psnr, ssim, runtime) in zip(pair_list, metrics):
            stem = hazy.stem; fog = "heavy" if "0.2" in stem else "medium" if any(x in stem for x in ("0.12", "0.16")) else "light"
            all_rows.append({"dataset": dataset, "data_origin": "synthetic" if dataset.startswith("sots") else "real", "fog_level": fog, "output_psnr": psnr, "output_ssim": ssim, "runtime_ms": runtime})
        total_time += inference_time; total_images += len(pair_list); print(f"{label}: {len(pair_list)} images, PSNR={np.mean([m[0] for m in metrics]):.3f}, SSIM={np.mean([m[1] for m in metrics]):.4f}")
    args.output_dir.mkdir(parents=True, exist_ok=True); paths = export_excel_reports("HazeWaveNet", all_rows, args.output_dir); write_hardware_report(args.output_dir, "HazeWaveNet", total_images, total_time, total_time)
    print("Saved:"); [print(path) for path in paths.values()]; print(args.output_dir / "hardware.xlsx")

if __name__ == "__main__": main()
