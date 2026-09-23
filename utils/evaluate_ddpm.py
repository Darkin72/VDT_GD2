"""Benchmark DehazeDDPM-NH with the common VDT-GD2 metrics and Excel reports."""

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from openpyxl import Workbook

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DDPM_ROOT = PROJECT_ROOT / "solution" / "DehazeDDPM"
sys.path.insert(0, str(PROJECT_ROOT / "utils"))
sys.path.insert(0, str(DDPM_ROOT))

from excel_report import export_excel_reports
from evaluate import (
    DATASET_LOADERS,
    calculate_psnr,
    calculate_ssim,
    infer_data_origin,
    infer_fog_level,
    resize_pair,
)


def load_config(path):
    text = "".join(line.split("//")[0] + "\n" for line in path.read_text().splitlines())
    return json.loads(text)


def load_model(args):
    import model as ddpm_model

    config = load_config(args.config)
    config["phase"] = "val"
    config["gpu_ids"] = [0] if args.device.startswith("cuda") else None
    config["distributed"] = False
    config["path"]["resume_state"] = str(args.checkpoint).removesuffix("_gen.pth")
    config["path"]["resume_stateH"] = str(args.prenet)
    config["path"]["log"] = str(args.output_dir / "logs")
    config["path"]["results"] = str(args.output_dir / "images")
    config["path"]["checkpoint"] = str(args.output_dir / "checkpoint")
    config["path"]["experiments_root"] = str(args.output_dir)
    config["model"]["diffusion"]["image_size"] = args.image_size
    model = ddpm_model.create_model(config)
    model.set_new_noise_schedule(config["model"]["beta_schedule"]["val"], schedule_phase="val")
    return model


def infer_one(model, image, device, image_size):
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(image.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)
    with torch.inference_mode():
        model.feed_data({"SR": tensor.to(device), "HR": tensor.to(device)})
        model.test(continous=True)
        output = model.get_current_visuals()["Out"][-1].permute(1, 2, 0).numpy()
    return (output.clip(0, 1) * 255).round().astype(np.uint8)


def parse_args():
    root = PROJECT_ROOT
    parser = argparse.ArgumentParser(description="Evaluate DehazeDDPM-NH on VDT-GD2 datasets.")
    parser.add_argument("--data-root", type=Path, default=root / "dataset")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--config", type=Path, default=DDPM_ROOT / "config" / "test_NH.json")
    parser.add_argument("--checkpoint", type=Path, default=DDPM_ROOT / "Diffusion_trained_pth" / "NH_I230000_E4600_gen.pth")
    parser.add_argument("--prenet", type=Path, default=DDPM_ROOT / "pretrained_PreNet_pth" / "NH_net_g_80000.pth")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=root / "utils" / "evaluation_results" / "dehazeddpm_nh")
    parser.add_argument("--save-images", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "images").mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    model = load_model(args)
    device = torch.device(args.device)
    rows = []
    pairs = []
    for dataset in DATASET_LOADERS:
        values = DATASET_LOADERS[dataset](args.data_root, args.split)
        pairs.extend(values[:args.limit] if args.limit > 0 else values)
    for dataset, image_id, hazy_path, clear_path in pairs:
        hazy = cv2.cvtColor(cv2.imread(str(hazy_path)), cv2.COLOR_BGR2RGB)
        clear = cv2.cvtColor(cv2.imread(str(clear_path)), cv2.COLOR_BGR2RGB)
        hazy, clear = resize_pair(hazy, clear, args.image_size)
        start = time.perf_counter()
        output = infer_one(model, hazy, device, args.image_size)
        runtime_ms = (time.perf_counter() - start) * 1000
        clear = cv2.resize(clear, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA)
        row = {
            "dataset": dataset, "image_id": image_id,
            "data_origin": infer_data_origin(dataset),
            "fog_level": infer_fog_level(dataset, image_id),
            "width": args.image_size, "height": args.image_size,
            "hazy_psnr": calculate_psnr(hazy, clear), "hazy_ssim": calculate_ssim(hazy, clear),
            "output_psnr": calculate_psnr(output, clear), "output_ssim": calculate_ssim(output, clear),
            "psnr_improvement": calculate_psnr(output, clear) - calculate_psnr(hazy, clear),
            "ssim_improvement": calculate_ssim(output, clear) - calculate_ssim(hazy, clear),
            "runtime_ms": runtime_ms, "hazy_path": str(hazy_path), "clear_path": str(clear_path),
        }
        rows.append(row)
        if args.save_images:
            cv2.imwrite(str(args.output_dir / "images" / f"{dataset}_{image_id}_dehazeddpm_nh.png"), cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
        print(dataset, image_id, f"PSNR={row['output_psnr']:.4f}", f"SSIM={row['output_ssim']:.4f}", f"ms={runtime_ms:.2f}")
    if not rows:
        raise RuntimeError("No paired images found.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_image.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    export_excel_reports("DehazeDDPM-NH", rows, args.output_dir)
    elapsed = sum(row["runtime_ms"] for row in rows) / 1000
    hardware = {"os": platform.platform(), "cpu": platform.processor() or platform.machine(), "cpu_count": os.cpu_count(), "device": args.device}
    try:
        import psutil
        process = psutil.Process(os.getpid())
        hardware["ram_total_gb"] = round(psutil.virtual_memory().total / 1024**3, 2)
        hardware["ram_end_gb"] = round(process.memory_info().rss / 1024**3, 2)
    except ImportError:
        hardware["ram"] = "psutil not installed"
    if args.device.startswith("cuda") and torch.cuda.is_available():
        index = torch.device(args.device).index or torch.cuda.current_device()
        hardware.update({"gpu": torch.cuda.get_device_name(index), "gpu_vram_total_gb": round(torch.cuda.get_device_properties(index).total_memory / 1024**3, 2), "gpu_vram_peak_allocated_gb": round(torch.cuda.max_memory_allocated(index) / 1024**3, 2), "gpu_vram_peak_reserved_gb": round(torch.cuda.max_memory_reserved(index) / 1024**3, 2)})
    performance = {"images": len(rows), "inference_seconds": elapsed, "inference_fps": len(rows) / elapsed if elapsed else 0, "mean_ms_per_image": elapsed * 1000 / len(rows), "hardware": hardware}
    (args.output_dir / "performance.json").write_text(json.dumps(performance, indent=2), encoding="utf-8")
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Hardware"
    sheet.append(("Solution", "DehazeDDPM-NH"))
    for key, value in performance["hardware"].items():
        sheet.append((key, value))
    sheet.append(("images", performance["images"]))
    sheet.append(("inference_seconds", performance["inference_seconds"]))
    sheet.append(("inference_fps", performance["inference_fps"]))
    sheet.append(("mean_ms_per_image", performance["mean_ms_per_image"]))
    workbook.save(args.output_dir / "dehazeddpm-nh_hardware.xlsx")
    print(json.dumps(performance, indent=2))


if __name__ == "__main__":
    main()
