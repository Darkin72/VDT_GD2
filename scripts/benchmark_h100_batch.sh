#!/usr/bin/env bash
set -Eeuo pipefail

IMAGE="${IMAGE:-darkin72/dehazewavelet:latest}"
DATA_ROOT="${DATA_ROOT:-/root/clearair-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/clearair-output/h100-benchmark}"
CANDIDATES="${CANDIDATES:-1 2 4 8 16 32 64 128 256 512}"
mkdir -p "$OUTPUT_ROOT"
echo "Image=$IMAGE"
echo "Dataset=$DATA_ROOT"
echo "Candidates=$CANDIDATES"

docker run --rm -i --gpus all --ipc=host --shm-size=16g \
  --mount "type=bind,src=$DATA_ROOT,dst=/workspace/ClearAIR/dataset,readonly" \
  --mount "type=bind,src=$OUTPUT_ROOT,dst=/workspace/ClearAIR/outputs" \
  "$IMAGE" python -u - "$CANDIDATES" <<'PY'
import json
import sys
import time
from pathlib import Path

import torch
from solution.wavelet_dehaze.model import HazeWaveNet, haze_wavelet_loss
from solution.wavelet_dehaze.run_its_ots import paired_root
from solution.wavelet_dehaze.train import PairedImages

candidates = [int(value) for value in " ".join(sys.argv[1:]).split()]
dataset = PairedImages(paired_root(Path('/workspace/ClearAIR/dataset/ITS')), size=256, augment=False)
probe = dataset[0]
results = []
print(f'[1/3] GPU={torch.cuda.get_device_name(0)} total_vram_gb={torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}', flush=True)
print('[2/3] Forward/backward benchmark at size=256, AMP=float16', flush=True)
for batch_size in candidates:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = HazeWaveNet().cuda().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    hazy = probe[0].unsqueeze(0).cuda().repeat(batch_size, 1, 1, 1)
    clear = probe[1].unsqueeze(0).cuda().repeat(batch_size, 1, 1, 1)
    row = {'batch_size': batch_size, 'status': 'failed'}
    try:
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                prediction = model(hazy)
                loss = haze_wavelet_loss(prediction, clear)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            prediction = model(hazy)
            loss = haze_wavelet_loss(prediction, clear)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        row.update(status='ok', seconds=seconds, samples_per_second=batch_size / seconds,
                   loss=float(loss.detach().cpu()),
                   allocated_gb=torch.cuda.max_memory_allocated() / 1024**3,
                   reserved_gb=torch.cuda.max_memory_reserved() / 1024**3,
                   total_vram_gb=torch.cuda.get_device_properties(0).total_memory / 1024**3)
        print(row, flush=True)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        row.update(status='oom', allocated_gb=torch.cuda.max_memory_allocated() / 1024**3,
                   total_vram_gb=torch.cuda.get_device_properties(0).total_memory / 1024**3)
        print(row, flush=True)
    finally:
        del model, optimizer, hazy, clear
        torch.cuda.empty_cache()
    results.append(row)

output = Path('/workspace/ClearAIR/outputs/batch_benchmark.json')
output.write_text(json.dumps(results, indent=2))
print(f'[3/3] Saved {output}', flush=True)
PY
