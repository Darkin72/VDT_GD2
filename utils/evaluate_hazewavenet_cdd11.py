"""Evaluate the three Drive HazeWaveNet checkpoints on CDD-11."""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from solution.wavelet_dehaze.model import HazeWaveNet

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CHECKPOINTS = {
    "i_haze": "haze_wavelet_i_haze.pt",
    "o_hazy": "haze_wavelet_o_hazy.pt",
    "sots_its": "haze_wavelet_sots_its.pt",
}
CATEGORIES = ("low", "haze", "rain", "snow", "low_haze", "low_rain", "low_snow", "haze_rain", "haze_snow", "low_haze_rain", "low_haze_snow")


def image_key(path: Path) -> str:
    value = re.sub(
        r"(^|[_ .-])(hazy|haze|clear|clean|gt|groundtruth|target|input|degraded)(?=$|[_ .-])",
        "_",
        path.stem.lower(),
    )
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def find_pairs(root: Path) -> list[tuple[str, Path, Path]]:
    if (root / "clear").is_dir() and any((root / category).is_dir() for category in CATEGORIES):
        clear_images = {path.stem: path for path in (root / "clear").iterdir() if path.is_file() and path.suffix.lower() in EXTENSIONS}
        pairs = []
        for category in CATEGORIES:
            category_dir = root / category
            if not category_dir.is_dir():
                raise FileNotFoundError(f"CDD-11_test missing category: {category}")
            for path in sorted(category_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in EXTENSIONS:
                    if path.stem not in clear_images:
                        raise FileNotFoundError(f"No clear image for {path}")
                    pairs.append((f"{category}/{path.stem}", path, clear_images[path.stem]))
        if not pairs:
            raise RuntimeError(f"No images found below {root}")
        return pairs
    hazy, clear = {}, {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        folders = {part.lower() for part in path.parts}
        if folders & {"hazy", "haze", "input", "degraded"}:
            hazy.setdefault(image_key(path), path)
        elif folders & {"clear", "clean", "gt", "groundtruth", "target"}:
            clear.setdefault(image_key(path), path)
    pairs = [(key, hazy[key], clear[key]) for key in sorted(hazy.keys() & clear.keys())]
    if not pairs:
        raise RuntimeError(f"No hazy/clear pairs found below {root}")
    return pairs


def load_model(checkpoint: Path, device: str) -> HazeWaveNet:
    model = HazeWaveNet(levels=4).to(device).eval()
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(state, dict):
        state = state.get("model", state.get("state_dict", state))
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model


def run_checkpoint(label, checkpoint, pairs, output_dir, device, max_side):
    model = load_model(checkpoint, device)
    rows = []
    for index, (image_id, hazy_path, clear_path) in enumerate(pairs, 1):
        started = time.perf_counter()
        with Image.open(hazy_path) as source:
            hazy_image = source.convert("RGB")
        with Image.open(clear_path) as source:
            clear_image = source.convert("RGB")
        if max_side and max(hazy_image.size) > max_side:
            scale = max_side / max(hazy_image.size)
            size = (round(hazy_image.width * scale), round(hazy_image.height * scale))
            hazy_image = hazy_image.resize(size, Image.Resampling.LANCZOS)
            clear_image = clear_image.resize(size, Image.Resampling.LANCZOS)
        hazy = np.asarray(hazy_image, dtype=np.float32) / 255.0
        clear = np.asarray(clear_image, dtype=np.uint8)
        tensor = torch.from_numpy(hazy).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.inference_mode():
            result = model(tensor)[0].permute(1, 2, 0).mul(255).clamp(0, 255).byte().cpu().numpy()
        image_path = output_dir / "images" / label / f"{image_id}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result).save(image_path)
        input_image = (hazy * 255).astype(np.uint8)
        rows.append({
            "checkpoint": label, "image_id": image_id,
            "category": image_id.split("/")[0] if "/" in image_id else "haze",
            "hazy_path": str(hazy_path), "clear_path": str(clear_path),
            "input_psnr": peak_signal_noise_ratio(clear, input_image, data_range=255),
            "input_ssim": structural_similarity(clear, input_image, channel_axis=2, data_range=255),
            "output_psnr": peak_signal_noise_ratio(clear, result, data_range=255),
            "output_ssim": structural_similarity(clear, result, channel_axis=2, data_range=255),
            "runtime_ms": (time.perf_counter() - started) * 1000,
        })
        if index == 1 or index % 10 == 0 or index == len(pairs):
            print(f"[{label}] {index}/{len(pairs)}", flush=True)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cdd11-test", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-side", type=int, default=512)
    args = parser.parse_args()
    pairs = find_pairs(args.cdd11_test)
    rows = []
    for label, filename in CHECKPOINTS.items():
        checkpoint = args.checkpoint_dir / filename
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        rows.extend(run_checkpoint(label, checkpoint, pairs, args.output_dir, args.device, args.max_side))
    per_image = pd.DataFrame(rows)
    summary = per_image.groupby("checkpoint", as_index=False).agg(
        images=("image_id", "count"), input_psnr=("input_psnr", "mean"),
        input_ssim=("input_ssim", "mean"), output_psnr=("output_psnr", "mean"),
        output_ssim=("output_ssim", "mean"), runtime_ms=("runtime_ms", "mean"),
    )
    summary["psnr_improvement"] = summary.output_psnr - summary.input_psnr
    summary["ssim_improvement"] = summary.output_ssim - summary.input_ssim
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_image.to_csv(args.output_dir / "per_image.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(args.output_dir / "cdd11_hazewavenet_checkpoints.xlsx") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        per_image.to_excel(writer, sheet_name="Per image", index=False)
    print(summary.round(4).to_string(index=False))
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
