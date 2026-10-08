"""Evaluate WDMamba on a standard benchmark, official paired test split, or CDD-11."""

from __future__ import annotations

import argparse
import csv
import importlib.machinery
import importlib.util
import json
import re
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


def infer_detail_blocks(state):
    """Infer the 4/6-block DE architecture from weights, even after renaming."""
    groups = {group: set() for group in (1, 2, 3)}
    for key in state:
        match = re.fullmatch(r"restoration_network\.DE\.g([123])\.gp\.(\d+)\.conv1\.weight", key)
        if match:
            groups[int(match[1])].add(int(match[2]))
    counts = {len(indices) for indices in groups.values()}
    if len(counts) != 1 or next(iter(counts)) not in (4, 6):
        raise ValueError(f"Cannot infer a supported WDMamba DE architecture: {groups}")
    blocks = next(iter(counts))
    if any(indices != set(range(blocks)) for indices in groups.values()):
        raise ValueError("WDMamba checkpoint has incomplete DE block indices")
    return blocks


def import_architecture():
    # The upstream package __init__ imports its training stack (losses, LPIPS,
    # dataloaders, etc.). In this inference subprocess, load only architecture
    # packages from the local checkout and avoid those training dependencies.
    for name, directory in (
        ("basicsr", WDMAMBA_ROOT / "basicsr"),
        ("basicsr.utils", WDMAMBA_ROOT / "basicsr" / "utils"),
        ("basicsr.archs", WDMAMBA_ROOT / "basicsr" / "archs"),
    ):
        if name not in sys.modules:
            spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
            spec.submodule_search_locations = [str(directory)]
            sys.modules[name] = importlib.util.module_from_spec(spec)
    try:
        from basicsr.archs.wavemamba_arch import WaveMamba
        from basicsr.archs.detail_enhance_net import DENet
    except Exception as exc:
        raise RuntimeError(
            "Cannot import the WDMamba architecture. Run the WDMamba runtime setup cell "
            "in Bench.ipynb to install compatible CUDA extensions in a Python 3.11 environment. "
            f"Current interpreter: {sys.executable} (Python {sys.version.split()[0]}). "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    return WaveMamba, DENet


def check_environment(device):
    print(f"Python: {sys.version.split()[0]} ({sys.executable})", flush=True)
    print(f"PyTorch: {torch.__version__}; CUDA build: {torch.version.cuda}; device: {device}", flush=True)
    if torch.device(device).type != "cuda":
        raise RuntimeError("The upstream WDMamba selective-scan extension requires a CUDA GPU. Select a GPU runtime.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this interpreter. Select a GPU runtime and run the WDMamba runtime setup cell.")
    import_architecture()
    # Importing a wheel alone does not prove its CUDA kernels can run on this GPU.
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    with torch.inference_mode():
        u = torch.zeros(1, 4, 16, device=device)
        a = -torch.ones(4, 16, device=device)
        bc = torch.ones(1, 16, 16, device=device)
        result = selective_scan_fn(u, torch.ones_like(u), a, bc, bc, delta_softplus=True)
        if not torch.isfinite(result).all().item():
            raise RuntimeError("WDMamba selective-scan CUDA check produced non-finite values")
    print("WDMamba architecture imports and selective-scan CUDA check: OK", flush=True)


def load_model(checkpoint: Path, device: str, de_blocks=None):
    WaveMamba, DENet = import_architecture()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        state = state.get("params", state.get("state_dict", state.get("model", state)))
    state = {key.removeprefix("module."): value for key, value in state.items()}
    detected_blocks = infer_detail_blocks(state)
    if de_blocks is not None and de_blocks != detected_blocks:
        raise ValueError(f"--de-blocks={de_blocks} conflicts with checkpoint architecture ({detected_blocks})")
    print(f"WDMamba detail-enhancement blocks: {detected_blocks}", flush=True)
    model = WaveMamba(in_chn=3, wf=16, n_l_blocks=[1, 2, 2, 4], ffn_scale=2.0)
    if detected_blocks == 4:
        model.restoration_network.DE = DENet(3, 4)
    model.load_state_dict(state, strict=True)
    model.de_blocks = detected_blocks
    return model.to(device).eval()


def find_paired_images(root, dataset):
    """Official paired layouts use hazy + GT/gt; normalized copies use clear."""
    hazy_dir = root / "hazy"
    clear_dir = next((root / name for name in ("clear", "GT", "gt") if (root / name).is_dir()), None)
    if not hazy_dir.is_dir() or clear_dir is None:
        raise FileNotFoundError(f"Expected {root}/hazy and clear, GT, or gt")
    clear = {p.stem.lower(): p for p in clear_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}
    pairs = []
    for hazy in sorted(hazy_dir.iterdir()):
        if not hazy.is_file() or hazy.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        stem = hazy.stem.lower()
        if dataset in {"nh-haze", "dense-haze", "o-hazy"}:
            candidates = [stem.replace("_hazy", "_gt"), stem.removesuffix("_hazy"), stem]
        elif dataset == "haze4k":
            candidates = [stem.split("_")[0], stem]
        else:
            candidates = [stem]
        match = next((clear[name] for name in candidates if name in clear), None)
        if match is None:
            raise FileNotFoundError(f"No GT image for {hazy}")
        pairs.append((dataset, hazy.stem, hazy, match))
    return pairs


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
    parser.add_argument("--dataset", choices=("i-haze", "o-hazy", "sots-indoor", "sots-outdoor", "cdd11",
                                              "nh-haze", "dense-haze", "haze4k", "reside6k"), default="i-haze")
    parser.add_argument("--paired-root", type=Path, help="Explicit test folder with hazy and GT/gt/clear")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--cdd11-test", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--check-environment", action="store_true", help="Check architecture imports and run a small CUDA kernel, without weights or data")
    parser.add_argument("--de-blocks", type=int, choices=(4, 6), default=None,
                        help="Validate DE block count; default is inferred from checkpoint tensors")
    parser.add_argument("--checkpoint-training-dataset", default="unspecified")
    parser.add_argument("--evaluation-protocol", choices=("matched-dataset", "cross-dataset"), default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--save-images", action="store_true")
    args = parser.parse_args()
    if not args.check_environment and (args.checkpoint is None or args.output_dir is None):
        parser.error("--checkpoint and --output-dir are required for inference")
    return args


def main():
    args = parse_args()
    if args.check_environment:
        check_environment(args.device)
        return
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Missing WDMamba checkpoint: {args.checkpoint}")
    if args.dataset == "cdd11":
        if args.cdd11_test is None:
            raise ValueError("--cdd11-test is required when --dataset cdd11")
        raw_pairs = find_cdd11_pairs(args.cdd11_test)
        pairs = [(args.dataset, image_id, hazy, clear, category) for image_id, hazy, clear, category in raw_pairs]
    else:
        if args.paired_root is not None:
            values = find_paired_images(args.paired_root, args.dataset)
        elif args.dataset in DATASET_LOADERS:
            values = DATASET_LOADERS[args.dataset](args.data_root, args.split)
        else:
            raise ValueError(f"--paired-root is required for {args.dataset}; point it to the test split")
        values = values[:args.limit] if args.limit > 0 else values
        pairs = [(dataset, image_id, hazy, clear, infer_fog_level(dataset, image_id)) for dataset, image_id, hazy, clear in values]
    if args.limit > 0 and args.dataset == "cdd11":
        pairs = pairs[:args.limit]
    if not pairs:
        raise RuntimeError("No paired images found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_images:
        (args.output_dir / "images").mkdir(parents=True, exist_ok=True)
    model = load_model(args.checkpoint, args.device, args.de_blocks)
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
            "data_origin": infer_data_origin(dataset),
            "fog_level": infer_fog_level(dataset, image_id),
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
    excel_paths = export_excel_reports("WDMamba", rows, args.output_dir)
    write_hardware_report(args.output_dir, "WDMamba", len(rows), time.perf_counter() - started, sum(row["runtime_ms"] for row in rows) / 1000)
    excel_paths["hardware"] = args.output_dir / "hardware.xlsx"
    summary = {
        "model": "WDMamba", "dataset": args.dataset, "images": len(rows),
        "mean_output_psnr": float(np.mean([row["output_psnr"] for row in rows])),
        "mean_output_ssim": float(np.mean([row["output_ssim"] for row in rows])),
        "checkpoint": str(args.checkpoint), "device": args.device,
        "de_blocks": model.de_blocks,
        "checkpoint_training_dataset": args.checkpoint_training_dataset,
        "evaluation_protocol": args.evaluation_protocol,
        "split": args.split, "max_side": args.max_side, "limit": args.limit,
        "paired_root": str(args.paired_root) if args.paired_root else None,
        "excel_reports": {key: str(path) for key, path in excel_paths.items()},
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Excel reports:", flush=True)
    for path in excel_paths.values():
        print(" -", path.resolve(), flush=True)


if __name__ == "__main__":
    main()
