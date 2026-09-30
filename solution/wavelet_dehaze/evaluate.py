"""Evaluate HazeWaveNet checkpoints and export one Excel workbook per dataset."""
from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import numpy as np
import openpyxl
import psutil
import torch
from openpyxl.styles import Font
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .model import HazeWaveNet
from .train import PairedImages


def load_image(path: Path, max_side: int) -> tuple[torch.Tensor, np.ndarray]:
    image = Image.open(path).convert("RGB")
    if max_side > 0 and max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    array = np.array(image, copy=True)
    tensor = torch.from_numpy(array).float().div(255).permute(2, 0, 1).unsqueeze(0)
    return tensor, array


def hardware_info(device: str) -> dict[str, object]:
    info: dict[str, object] = {
        "Platform": platform.platform(), "Processor": platform.processor(),
        "CPU cores physical": psutil.cpu_count(logical=False), "CPU cores logical": psutil.cpu_count(),
        "RAM total (GB)": round(psutil.virtual_memory().total / 1024**3, 2), "Device": device,
    }
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        info.update({"GPU": props.name, "VRAM total (GB)": round(props.total_memory / 1024**3, 2),
                     "VRAM peak (GB)": round(torch.cuda.max_memory_allocated() / 1024**3, 3)})
    return info


def evaluate_dataset(name: str, root: Path, checkpoint: Path, output: Path, device: str, max_side: int) -> None:
    pairs = PairedImages(root, size=256).items
    model = HazeWaveNet().to(device).eval()
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state.get("model", state))
    rows, inference_total = [], 0.0
    if device == "cuda": torch.cuda.reset_peak_memory_stats()
    for hazy_path, clear_path in pairs:
        hazy, _ = load_image(hazy_path, max_side)
        clear_tensor, clear = load_image(clear_path, max_side)
        if hazy.shape[-2:] != clear_tensor.shape[-2:]:
            hazy = torch.nn.functional.interpolate(hazy, clear_tensor.shape[-2:], mode="bilinear", align_corners=False)
        hazy = hazy.to(device)
        if device == "cuda": torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode(): pred = model(hazy)
        if device == "cuda": torch.cuda.synchronize()
        runtime = (time.perf_counter() - start) * 1000
        inference_total += runtime
        result = pred[0].permute(1, 2, 0).mul(255).round().byte().cpu().numpy()
        rows.append((hazy_path.name, peak_signal_noise_ratio(clear, result, data_range=255),
                     structural_similarity(clear, result, channel_axis=2, data_range=255), runtime))
        print(f"{name}: {hazy_path.name} PSNR={rows[-1][1]:.3f} SSIM={rows[-1][2]:.4f} ms={runtime:.2f}")

    workbook = openpyxl.Workbook(); results = workbook.active; results.title = "Results"
    results.append(("Image", "PSNR (dB)", "SSIM", "Inference time (ms)"))
    for row in rows: results.append(row)
    results.append(("AVERAGE", float(np.mean([r[1] for r in rows])), float(np.mean([r[2] for r in rows])), float(np.mean([r[3] for r in rows]))))
    for cell in results[1]: cell.font = Font(bold=True)
    results.column_dimensions["A"].width = 30
    for column in "BCD": results.column_dimensions[column].width = 20

    summary = workbook.create_sheet("Summary")
    summary.append(("Dataset", name)); summary.append(("Images", len(rows)))
    summary.append(("Average PSNR (dB)", float(np.mean([r[1] for r in rows]))))
    summary.append(("Average SSIM", float(np.mean([r[2] for r in rows]))))
    summary.append(("Average inference (ms/image)", float(np.mean([r[3] for r in rows]))))
    summary.append(("FPS", 1000.0 / np.mean([r[3] for r in rows])))
    summary.append(("Total inference time (s)", inference_total / 1000.0))

    hardware = workbook.create_sheet("Hardware")
    hardware.append(("Property", "Value"))
    for key, value in hardware_info(device).items(): hardware.append((key, value))
    for sheet in (summary, hardware):
        sheet.column_dimensions["A"].width = 34; sheet.column_dimensions["B"].width = 50
        for cell in sheet[1]: cell.font = Font(bold=True)
    output.parent.mkdir(parents=True, exist_ok=True); workbook.save(output)
    print(f"Saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True); parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--name", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda"); parser.add_argument("--max-side", type=int, default=1024)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA is not available")
    evaluate_dataset(args.name, Path(args.dataset), args.checkpoint, args.output, args.device, args.max_side)


if __name__ == "__main__": main()
