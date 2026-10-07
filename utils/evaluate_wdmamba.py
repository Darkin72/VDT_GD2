"""Evaluate WDMamba on one VDT-GD2 dataset or CDD-11."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WDMAMBA_ROOT = PROJECT_ROOT / "solution" / "WDMamba"
sys.path.insert(0, str(PROJECT_ROOT / "utils"))
sys.path.insert(0, str(WDMAMBA_ROOT))

from evaluate import (  # noqa: E402
    DATASET_LOADERS,
    calculate_psnr,
    calculate_ssim,
    infer_data_origin,
    infer_fog_level,
    resize_pair,
)
from excel_report import export_excel_reports  # noqa: E402
from evaluation_hardware import write_hardware_report  # noqa: E402


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CDD11_CATEGORIES = (
    "low", "haze", "rain", "snow", "low_haze", "low_rain", "low_snow",
    "haze_rain", "haze_snow", "low_haze_rain", "low_haze_snow",
)


def find_cdd11_pairs(root: Path):
    clear_dir = root / "clear"
    if not clear_dir.is_dir():
        raise FileNotFoundError(f"Missing CDD-11 clear directory: {clear_dir}")
    clear = {
        path.stem: path
        for path in clear_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    pairs = []
    for category in CDD11_CATEGORIES:
        category_dir = root / category
        if not category_dir.is_dir():
            raise FileNotFoundError(f"Missing CDD-11 category directory: {category_dir}")
        for hazy in sorted(category_dir.iterdir()):
            if not hazy.is_file() or hazy.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if hazy.stem not in clear:
                raise FileNotFoundError(f"No clear image for {hazy}")
            pairs.append((f"{category}/{hazy.stem}", hazy, clear[hazy.stem], category))
    if not pairs:
        raise RuntimeError(f"No CDD-11 pairs found below {root}")
    return pairs


def load_model(checkpoint: Path, device: str):
    try:
        from basicsr.archs.wavemamba_arch import WaveMamba
    except ImportError as exc:
        raise RuntimeError(
            "WDMamba dependencies are missing. Install solution/WDMamba/requirements.txt "
            "and the compatible mamba_ssm/causal_conv1d packages."
        ) from exc
    model = WaveMamba(in_chn=3, wf=16, n_l_blocks=[1, 2, 2, 4], ffn_scale=2.0)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(state, dict):
        state = state.get("params", state.get("state_dict", state.get("model", state)))
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def infer_one(model, image, device, max_side):
    image, _ = resize_pair(image, image, max_side)
    height, width = image.shape[:2]
    tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    pad_h = (16 - height % 16) % 16
    pad_w = (16 - width % 16) % 16
    if pad_h or pad_w:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    with torch.inference_mode():
        output = model.restoration_network(tensor.to(device))
    output = output[..., :height, :width]
    return (output[0].permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "dataset")
    parser.add_argument("--dataset", choices=("i-haze", "o-hazy", "sots-indoor", "sots-outdoor", "cdd11"), default="i-haze")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--cdd11-test", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-images", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Missing WDMamba checkpoint: {args.checkpoint}")
    if args.dataset == "cdd11":
        if args.cdd11_test is None:
            raise ValueError("--cdd11-test is required when --dataset cdd11")
        raw_pairs = find_cdd11_pairs(args.cdd11_test)
        pairs = [(args.dataset, image_id, hazy, clear, category) for image_id, hazy, clear, category in raw_pairs]
    else:
        values = DATASET_LOADERS[args.dataset](args.data_root, args.split)
        values = values[:args.limit] if args.limit > 0 else values
        pairs = [(dataset, image_id, hazy, clear, infer_fog_level(dataset, image_id)) for dataset, image_id, hazy, clear in values]
    if args.limit > 0 and args.dataset == "cdd11":
        pairs = pairs[:args.limit]
    if not pairs:
        raise RuntimeError("No paired images found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_images:
        (args.output_dir / "images").mkdir(parents=True, exist_ok=True)
    model = load_model(args.checkpoint, args.device)
    rows = []
    started = time.perf_counter()
    for index, (dataset, image_id, hazy_path, clear_path, category) in enumerate(pairs, 1):
        hazy_bgr = cv2.imread(str(hazy_path), cv2.IMREAD_COLOR)
        clear_bgr = cv2.imread(str(clear_path), cv2.IMREAD_COLOR)
        if hazy_bgr is None or clear_bgr is None:
            raise RuntimeError(f"Cannot read pair: {hazy_path}, {clear_path}")
        hazy = cv2.cvtColor(hazy_bgr, cv2.COLOR_BGR2RGB)
        clear = cv2.cvtColor(clear_bgr, cv2.COLOR_BGR2RGB)
        hazy, clear = resize_pair(hazy, clear, args.max_side)
        infer_started = time.perf_counter()
        output = infer_one(model, hazy, args.device, 0)
        runtime_ms = (time.perf_counter() - infer_started) * 1000
        row = {
            "dataset": dataset, "category": category, "image_id": image_id,
            "input_psnr": calculate_psnr(hazy, clear), "input_ssim": calculate_ssim(hazy, clear),
            "output_psnr": calculate_psnr(output, clear), "output_ssim": calculate_ssim(output, clear),
            "psnr_improvement": calculate_psnr(output, clear) - calculate_psnr(hazy, clear),
            "ssim_improvement": calculate_ssim(output, clear) - calculate_ssim(hazy, clear),
            "runtime_ms": runtime_ms, "hazy_path": str(hazy_path), "clear_path": str(clear_path),
        }
        rows.append(row)
        if args.save_images:
            safe_id = image_id.replace("/", "_").replace("\\", "_")
            cv2.imwrite(str(args.output_dir / "images" / f"{safe_id}_wdmamba.png"), cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
        print(f"[{index}/{len(pairs)}] {image_id} PSNR={row['output_psnr']:.4f} SSIM={row['output_ssim']:.4f} ms={runtime_ms:.2f}", flush=True)

    with (args.output_dir / "per_image.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    export_excel_reports("WDMamba", rows, args.output_dir)
    write_hardware_report(args.output_dir, "WDMamba", len(rows), time.perf_counter() - started, sum(row["runtime_ms"] for row in rows) / 1000)
    summary = {
        "model": "WDMamba", "dataset": args.dataset, "images": len(rows),
        "mean_output_psnr": float(np.mean([row["output_psnr"] for row in rows])),
        "mean_output_ssim": float(np.mean([row["output_ssim"] for row in rows])),
        "checkpoint": str(args.checkpoint), "device": args.device,
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
