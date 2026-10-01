#!/usr/bin/env bash
set -Eeuo pipefail

IMAGE="${IMAGE:-darkin72/dehazewavelet:latest}"
DATA_ROOT="${DATA_ROOT:-/root/clearair-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/clearair-output}"

for required_dir in ITS OTS; do
  if [[ ! -d "$DATA_ROOT/$required_dir" ]]; then
    echo "Missing training dataset: $DATA_ROOT/$required_dir" >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_ROOT"
echo "Image: $IMAGE"
echo "Dataset: $DATA_ROOT"
echo "Output: $OUTPUT_ROOT"

docker run --rm -i \
  --gpus all \
  --ipc=host \
  --shm-size=8g \
  --mount "type=bind,src=$DATA_ROOT,dst=/workspace/ClearAIR/dataset,readonly" \
  --mount "type=bind,src=$OUTPUT_ROOT,dst=/workspace/ClearAIR/outputs" \
  "$IMAGE" \
  python -u - <<'PY'
import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

from solution.wavelet_dehaze.evaluate import pairs
from solution.wavelet_dehaze.run_its_ots import paired_root
from solution.wavelet_dehaze.train import PairedImages

data_root = Path("/workspace/ClearAIR/dataset")
output_root = Path("/workspace/ClearAIR/outputs/smoke-train")
input_root = output_root / "input"
results_root = output_root / "results"
output_root.mkdir(parents=True, exist_ok=True)


def make_train_dataset(name, limit=4):
    source = paired_root(data_root / name)
    dataset = PairedImages(source)
    selected = []
    seen_clear = set()
    for hazy_path, clear_path in dataset.items:
        if clear_path not in seen_clear:
            selected.append((hazy_path, clear_path))
            seen_clear.add(clear_path)
        if len(selected) >= limit:
            break
    if len(selected) < 2:
        raise RuntimeError(f"Not enough pairs in {source}")
    destination = input_root / name
    for kind in ("hazy", "clear"):
        (destination / kind).mkdir(parents=True, exist_ok=True)
    for index, (hazy_path, clear_path) in enumerate(selected):
        for kind, source_path in (("hazy", hazy_path), ("clear", clear_path)):
            target = destination / kind / f"{index:04d}.png"
            with Image.open(source_path) as image:
                image.convert("RGB").resize((64, 64)).save(target)
    print(f"{name}: {len(selected)} train pairs", flush=True)
    return destination


def make_eval_dataset(label, source_root, dataset_name):
    matched = pairs(source_root, dataset_name)
    if not matched:
        raise RuntimeError(f"No evaluation pair for {label} under {source_root}")
    hazy_path, clear_path = matched[0]
    if label == "sots-indoor":
        destination = output_root / "eval" / "Synthetic Objective Testing Set (SOTS) [RESIDE]" / "indoor" / "test"
    elif label == "sots-outdoor":
        destination = output_root / "eval" / "Synthetic Objective Testing Set (SOTS) [RESIDE]" / "outdoor" / "test"
    else:
        destination = output_root / "eval" / ("I-HAZE" if label == "i-haze" else "O-HAZY") / "test"
    for kind in ("hazy", "clear"):
        (destination / kind).mkdir(parents=True, exist_ok=True)
    shutil.copy2(hazy_path, destination / "hazy" / hazy_path.name)
    shutil.copy2(clear_path, destination / "clear" / clear_path.name)
    print(f"{label}: evaluation pair copied", flush=True)


print("[1/5] Creating tiny train and evaluation datasets", flush=True)
its_data = make_train_dataset("ITS")
ots_data = make_train_dataset("OTS")
sots_root = data_root / "Synthetic Objective Testing Set (SOTS) [RESIDE]"
make_eval_dataset("i-haze", data_root / "I-HAZE", "i-haze")
make_eval_dataset("o-hazy", data_root / "O-HAZY", "o-hazy")
make_eval_dataset("sots-indoor", sots_root, "sots-indoor")
make_eval_dataset("sots-outdoor", sots_root, "sots-outdoor")

print("[2/5] Training ITS and OTS for one epoch", flush=True)
subprocess.run([
    sys.executable, "-m", "solution.wavelet_dehaze.run_its_ots",
    "--its-data", str(its_data),
    "--ots-data", str(ots_data),
    "--eval-data-root", str(output_root / "eval"),
    "--output-dir", str(results_root),
    "--epochs", "1",
    "--batch-size", "1",
    "--micro-batch-size", "1",
    "--size", "64",
    "--device", "cuda",
    "--num-workers", "0",
    "--val-fraction", "0.25",
    "--no-persistent-workers",
    "--amp",
], check=True)

print("[3/5] Checking checkpoints, histories and plots", flush=True)
expected = [
    "haze_wavelet_its.pt", "haze_wavelet_ots.pt",
    "history_its.json", "history_ots.json",
    "its_loss.png", "its_psnr.png", "its_ssim.png",
    "ots_loss.png", "ots_psnr.png", "ots_ssim.png",
    "reports/hazewavenet_dataset.xlsx",
    "reports/hazewavenet_domain.xlsx",
    "reports/hazewavenet_fog.xlsx",
    "reports/hardware.xlsx",
]
for relative_path in expected:
    path = results_root / relative_path
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty artifact: {path}")
    print(f"PASS {relative_path}: {path.stat().st_size} bytes", flush=True)

for name in ("its", "ots"):
    records = json.loads((results_root / f"history_{name}.json").read_text())
    if len(records) != 1:
        raise RuntimeError(f"Expected one history record for {name}")
    print(f"PASS history_{name}: {records[-1]}", flush=True)

print("[4/5] Checking inference", flush=True)
inference = results_root / "inference_smoke.png"
subprocess.run([
    sys.executable, "-m", "solution.wavelet_dehaze.infer",
    str(its_data / "hazy/0000.png"), str(inference),
    "--checkpoint", str(results_root / "haze_wavelet_its.pt"),
], check=True)
with Image.open(inference) as image:
    image.verify()
print(f"PASS inference_smoke.png: {inference.stat().st_size} bytes", flush=True)

print("[5/5] SMOKE TRAIN: PASS", flush=True)
print(f"Results: {results_root}", flush=True)
PY

echo "Smoke training completed: $OUTPUT_ROOT/smoke-train/results"
