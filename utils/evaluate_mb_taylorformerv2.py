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
from evaluation_hardware import write_hardware_report
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


def infer_batch(model, images, device):
    shapes = [image.shape[:2] for image in images]
    target_h = (max(height for height, _ in shapes) + 7) // 8 * 8
    target_w = (max(width for _, width in shapes) + 7) // 8 * 8
    tensors = []
    for image, (height, width) in zip(images, shapes):
        tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)
        tensors.append(F.pad(tensor, (0, target_w - width, 0, target_h - height), mode="replicate"))
    batch = torch.stack(tensors).to(device)
    with torch.inference_mode():
        outputs = model(batch).cpu()
    return [
        (outputs[index, :, :height, :width].permute(1, 2, 0).numpy().clip(0, 1) * 255).round().astype(np.uint8)
        for index, (height, width) in enumerate(shapes)
    ]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "dataset")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--size", choices=("B", "L"), default="B")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "mb-taylorformerv2" / "OTS-B.pth")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-side", type=int, default=0, help="Optional whole-image resize; 0 keeps original resolution.")
    parser.add_argument("--batch-size", type=int, default=8)
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
    inference_start = time.perf_counter()
    for offset in range(0, len(pairs), args.batch_size):
        batch_pairs = pairs[offset:offset + args.batch_size]
        prepared = []
        for item in batch_pairs:
            dataset, image_id, hazy_path, clear_path = item
            hazy = cv2.cvtColor(cv2.imread(str(hazy_path)), cv2.COLOR_BGR2RGB)
            clear = cv2.cvtColor(cv2.imread(str(clear_path)), cv2.COLOR_BGR2RGB)
            hazy, clear = resize_pair(hazy, clear, args.max_side)
            prepared.append((item, hazy, clear))
        if args.device.startswith("cuda"): torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = infer_batch(model, [item[1] for item in prepared], args.device)
        if args.device.startswith("cuda"): torch.cuda.synchronize()
        runtime_ms = (time.perf_counter() - start) * 1000 / len(prepared)
        for ((dataset, image_id, hazy_path, clear_path), hazy, clear), output in zip(prepared, outputs):
            row = {"dataset": dataset, "image_id": image_id, "data_origin": infer_data_origin(dataset), "fog_level": infer_fog_level(dataset, image_id), "width": clear.shape[1], "height": clear.shape[0], "hazy_psnr": calculate_psnr(hazy, clear), "hazy_ssim": calculate_ssim(hazy, clear), "output_psnr": calculate_psnr(output, clear), "output_ssim": calculate_ssim(output, clear), "psnr_improvement": calculate_psnr(output, clear) - calculate_psnr(hazy, clear), "ssim_improvement": calculate_ssim(output, clear) - calculate_ssim(hazy, clear), "runtime_ms": runtime_ms, "hazy_path": str(hazy_path), "clear_path": str(clear_path)}
            rows.append(row)
            if args.save_images: cv2.imwrite(str(image_dir / f"{dataset}_{image_id}_mb_taylorformerv2.png"), cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
            print(dataset, image_id, f"PSNR={row['output_psnr']:.4f}", f"SSIM={row['output_ssim']:.4f}", f"ms={runtime_ms:.2f}")
    inference_seconds = time.perf_counter() - inference_start
    if not rows:
        raise RuntimeError("No paired images found.")
    with (args.output_dir / "per_image.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    export_excel_reports("MB-TaylorFormerV2", rows, args.output_dir)
    write_hardware_report(args.output_dir, "MB-TaylorFormerV2", len(rows), inference_seconds, inference_seconds)


if __name__ == "__main__":
    main()
