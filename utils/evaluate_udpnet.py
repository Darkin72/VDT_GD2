"""Evaluate a UDPNet dehazing checkpoint on the common VDT-GD2 test pairs."""

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
UDP_ROOT = PROJECT_ROOT / "solution" / "UDPNet" / "Dehazing" / "OTS"
sys.path.insert(0, str(PROJECT_ROOT / "utils"))
sys.path.insert(0, str(UDP_ROOT))
from evaluate import DATASET_LOADERS, calculate_psnr, calculate_ssim, infer_data_origin, infer_fog_level, resize_pair
from excel_report import export_excel_reports
from evaluation_hardware import write_hardware_report


def load_model(args, checkpoint_path):
    module_name = "models.FSNet_UDPNet" if args.model == "FSNet" else "models.ConvIR_UDPNet"
    module = __import__(module_name, fromlist=["build_net"])
    model = module.build_net()
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    state = checkpoint.get("state_dict", checkpoint.get("params", checkpoint))
    state = {key.replace("model.", "", 1).removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=False)
    return model.to(args.device).eval()


def checkpoint_map(args):
    base = args.checkpoint_dir
    if base.suffix.lower() in {".pth", ".ckpt"}:
        base = base.parent
    prefix = f"{args.model}_UDPNet_"
    return {
        "sots-indoor": base / f"{prefix}ITS.ckpt",
        "sots-outdoor": base / f"{prefix}OTS.ckpt",
        "o-hazy": base / f"{prefix}OTS.ckpt",
        "i-haze": base / f"{prefix}ITS.ckpt",
    }


def load_depth_model(model_name, device):
    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ImportError as exc:
        raise RuntimeError("Depth Anything V2 requires transformers. Run: pip install -U transformers") from exc
    # The original HF repo only contains a raw .pth checkpoint. Use its
    # Transformers-converted counterpart when the original ID is supplied.
    if model_name == "depth-anything/Depth-Anything-V2-Base":
        model_name = "depth-anything/Depth-Anything-V2-Base-hf"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device).eval()
    return processor, model


def depth_channels(images, depth_paths, depth_pipeline):
    if depth_pipeline is not None:
        processor, model, device = depth_pipeline
        pixel_values = [processor(images=image, return_tensors="pt")["pixel_values"][0] for image in images]
        target_h = max(value.shape[1] for value in pixel_values)
        target_w = max(value.shape[2] for value in pixel_values)
        pixel_values = [F.pad(value, (0, target_w - value.shape[2], 0, target_h - value.shape[1])) for value in pixel_values]
        inputs = {"pixel_values": torch.stack(pixel_values).to(device)}
        with torch.inference_mode():
            predictions = model(**inputs).predicted_depth
        depths = []
        for prediction, image in zip(predictions, images):
            depth = F.interpolate(prediction[None, None], size=image.shape[:2], mode="bilinear", align_corners=False).squeeze().cpu().numpy().astype(np.float32)
            depth -= depth.min(); maximum = depth.max()
            depths.append(depth / maximum if maximum > 1e-8 else np.zeros_like(depth))
        return depths
    depths = []
    for image, depth_path in zip(images, depth_paths):
        depth = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE) if depth_path and depth_path.exists() else None
        if depth is None:
            raise FileNotFoundError(f"Depth map not found: {depth_path}")
        depths.append(cv2.resize(depth, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0)
    return depths


def depth_channel(image, depth_path, depth_pipeline):
    if depth_path is not None and depth_path.exists():
        depth = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
        if depth is not None:
            depth = cv2.resize(depth, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
            return depth.astype(np.float32) / 255.0
    if depth_pipeline is None:
        raise FileNotFoundError(f"Depth map not found: {depth_path}. Either provide valid maps or omit --depth-dir to use Depth Anything V2.")
    processor, model, device = depth_pipeline
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        prediction = model(**inputs).predicted_depth
    prediction = F.interpolate(prediction.unsqueeze(1), size=image.shape[:2], mode="bilinear", align_corners=False)
    depth = prediction.squeeze().cpu().numpy().astype(np.float32)
    depth -= depth.min()
    maximum = depth.max()
    return depth / maximum if maximum > 1e-8 else np.zeros_like(depth)


def infer_batch(model, images, depth_paths, device, depth_pipeline):
    shapes = [image.shape[:2] for image in images]
    depths = depth_channels(images, depth_paths, depth_pipeline)
    target_h = (max(height for height, _ in shapes) + 7) // 8 * 8
    target_w = (max(width for _, width in shapes) + 7) // 8 * 8
    tensors = []
    for image, depth, (height, width) in zip(images, depths, shapes):
        source = torch.from_numpy(np.concatenate([image.astype(np.float32) / 255.0, depth[..., None]], axis=2)).permute(2, 0, 1)
        tensors.append(F.pad(source, (0, target_w - width, 0, target_h - height), mode="replicate"))
    batch = torch.stack(tensors).to(device)
    with torch.inference_mode():
        outputs = model(batch)[-1].cpu()
    return [(outputs[index, :, :height, :width].permute(1, 2, 0).numpy().clip(0, 1) * 255).round().astype(np.uint8) for index, (height, width) in enumerate(shapes)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "dataset")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--model", choices=("FSNet", "ConvIR"), default="FSNet")
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "UDPNet_checkpoints")
    parser.add_argument("--depth-dir", type=Path, default=None, help="Optional directory containing depth maps named like hazy images.")
    parser.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Base-hf", help="Hugging Face Transformers Depth Anything V2 model ID.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-side", type=int, default=0, help="Optional whole-image resize. 0 preserves the original resolution.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "utils" / "evaluation_results" / "udpnet")
    parser.add_argument("--save-images", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True); (args.output_dir / "images").mkdir(exist_ok=True)
    depth_pipeline = None if args.depth_dir else (*load_depth_model(args.depth_model, args.device), args.device)
    if args.depth_dir is None:
        print(f"Using Depth Anything V2 depth model: {args.depth_model}")
    pairs = []
    for dataset in DATASET_LOADERS:
        values = DATASET_LOADERS[dataset](args.data_root, args.split); pairs.extend(values[:args.limit] if args.limit > 0 else values)
    rows = []
    inference_start = time.perf_counter()
    model_inference_seconds = 0.0
    models = {}
    checkpoints = checkpoint_map(args)
    for dataset_name in DATASET_LOADERS:
        dataset_pairs = [item for item in pairs if item[0] == dataset_name]
        for offset in range(0, len(dataset_pairs), args.batch_size):
            batch_pairs = dataset_pairs[offset:offset + args.batch_size]; prepared = []
            checkpoint_path = checkpoints[dataset_name]
            if dataset_name not in models:
                if not checkpoint_path.exists(): raise FileNotFoundError(f"Missing checkpoint for {dataset_name}: {checkpoint_path}")
                models[dataset_name] = load_model(args, checkpoint_path)
            for item in batch_pairs:
                dataset, image_id, hazy_path, clear_path = item
                hazy = cv2.cvtColor(cv2.imread(str(hazy_path)), cv2.COLOR_BGR2RGB); clear = cv2.cvtColor(cv2.imread(str(clear_path)), cv2.COLOR_BGR2RGB)
                hazy, clear = resize_pair(hazy, clear, args.max_side); prepared.append((item, hazy, clear))
            depth_paths = [args.depth_dir / item[0][2].name if args.depth_dir else None for item in prepared]
            if args.device.startswith("cuda"): torch.cuda.synchronize()
            start = time.perf_counter(); outputs = infer_batch(models[dataset_name], [item[1] for item in prepared], depth_paths, args.device, depth_pipeline)
            if args.device.startswith("cuda"): torch.cuda.synchronize()
            runtime_ms = (time.perf_counter() - start) * 1000 / len(prepared); model_inference_seconds += (runtime_ms / 1000.0) * len(prepared)
            for ((dataset, image_id, hazy_path, clear_path), hazy, clear), output in zip(prepared, outputs):
                hazy_psnr, output_psnr = calculate_psnr(hazy, clear), calculate_psnr(output, clear); hazy_ssim, output_ssim = calculate_ssim(hazy, clear), calculate_ssim(output, clear)
                row = {"dataset": dataset, "image_id": image_id, "data_origin": infer_data_origin(dataset), "fog_level": infer_fog_level(dataset, image_id), "width": clear.shape[1], "height": clear.shape[0], "hazy_psnr": hazy_psnr, "hazy_ssim": hazy_ssim, "output_psnr": output_psnr, "output_ssim": output_ssim, "psnr_improvement": output_psnr - hazy_psnr, "ssim_improvement": output_ssim - hazy_ssim, "runtime_ms": runtime_ms, "hazy_path": str(hazy_path), "clear_path": str(clear_path)}
                rows.append(row); print(dataset, image_id, f"PSNR={output_psnr:.4f}", f"SSIM={output_ssim:.4f}", f"ms={runtime_ms:.2f}")
                if args.save_images: cv2.imwrite(str(args.output_dir / "images" / f"{dataset}_{image_id}_udpnet.png"), cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    if not rows: raise RuntimeError("No paired images found.")
    with (args.output_dir / "per_image.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    export_excel_reports("UDPNet", rows, args.output_dir)
    write_hardware_report(args.output_dir, "UDPNet", len(rows), time.perf_counter() - inference_start, model_inference_seconds)


if __name__ == "__main__": main()
