"""Evaluate MB-TaylorFormerV2 on the common VDT-GD2 test pairs."""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MB_ROOT = PROJECT_ROOT / "solution" / "MB-TaylorFormerV2"
sys.path.insert(0, str(PROJECT_ROOT / "utils"))
sys.path.insert(0, str(MB_ROOT))

from evaluate import DATASET_LOADERS, calculate_psnr, calculate_ssim, infer_data_origin, infer_fog_level, resize_pair
from excel_report import export_excel_reports
from basicsr.models.archs.MB_TaylorFormerV2 import MB_TaylorFormer


def load_model(args):
    import yaml

    config_path = MB_ROOT / "Dehazing" / "Options" / f"MB-TaylorFormerV2-{args.size}.yml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["network_g"].pop("type", None)
    model = MB_TaylorFormer(**config["network_g"])
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    state = checkpoint.get("params", checkpoint.get("state_dict", checkpoint))
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=False)
    return model.to(args.device).eval()


def infer_one(model, image, device):
    height, width = image.shape[:2]
    tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    pad_h = (8 - height % 8) % 8
    pad_w = (8 - width % 8) % 8
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    with torch.inference_mode():
        output = model(tensor)[:, :, :height, :width]
    return (output.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "dataset")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--size", choices=("B", "L"), default="B")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "mb-taylorformerv2" / "OTS-B.pth")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-side", type=int, default=0, help="Resize pairs; 0 keeps original resolution.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "utils" / "evaluation_results" / "mb_taylorformerv2")
    parser.add_argument("--save-images", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.output_dir / "images"
    image_dir.mkdir(exist_ok=True)
    model = load_model(args)
    pairs = []
    for dataset in DATASET_LOADERS:
        values = DATASET_LOADERS[dataset](args.data_root, args.split)
        pairs.extend(values[:args.limit] if args.limit > 0 else values)
    rows = []
    for dataset, image_id, hazy_path, clear_path in pairs:
        hazy = cv2.cvtColor(cv2.imread(str(hazy_path)), cv2.COLOR_BGR2RGB)
        clear = cv2.cvtColor(cv2.imread(str(clear_path)), cv2.COLOR_BGR2RGB)
        hazy, clear = resize_pair(hazy, clear, args.max_side)
        start = time.perf_counter()
        output = infer_one(model, hazy, args.device)
        runtime_ms = (time.perf_counter() - start) * 1000
        row = {"dataset": dataset, "image_id": image_id, "data_origin": infer_data_origin(dataset), "fog_level": infer_fog_level(dataset, image_id), "width": clear.shape[1], "height": clear.shape[0], "hazy_psnr": calculate_psnr(hazy, clear), "hazy_ssim": calculate_ssim(hazy, clear), "output_psnr": calculate_psnr(output, clear), "output_ssim": calculate_ssim(output, clear), "psnr_improvement": calculate_psnr(output, clear) - calculate_psnr(hazy, clear), "ssim_improvement": calculate_ssim(output, clear) - calculate_ssim(hazy, clear), "runtime_ms": runtime_ms, "hazy_path": str(hazy_path), "clear_path": str(clear_path)}
        rows.append(row)
        if args.save_images:
            cv2.imwrite(str(image_dir / f"{dataset}_{image_id}_mb_taylorformerv2.png"), cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
        print(dataset, image_id, f"PSNR={row['output_psnr']:.4f}", f"SSIM={row['output_ssim']:.4f}", f"ms={runtime_ms:.2f}")
    if not rows:
        raise RuntimeError("No paired images found.")
    with (args.output_dir / "per_image.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    export_excel_reports("MB-TaylorFormerV2", rows, args.output_dir)


if __name__ == "__main__":
    main()
