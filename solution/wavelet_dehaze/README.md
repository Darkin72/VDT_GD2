# HazeWaveNet (wavelet dehazing)

This folder implements the method in *Lightweight single-image dehazing via haze-aware attention and discrete wavelet transform*. It uses fixed four-level Haar DWT, DCP-derived transmittance/trend guidance, guided low-frequency blocks, lightweight high-frequency processing, and learnable WIM fusion.

```powershell
python -m solution.wavelet_dehaze.infer input.png output.png
python -m solution.wavelet_dehaze.infer input.png output.png --checkpoint checkpoint.pt
python -m solution.wavelet_dehaze.train dataset/I-HAZE/train --epochs 50 --out haze_wavelet.pt
python -m solution.wavelet_dehaze.train dataset/I-HAZE/train --device cpu --epochs 1 --size 64
```

The training directory must contain paired files under `hazy/` and `clear/`.
Names may match directly, or the hazy image may use an `_hazy` suffix, for example
`hazy/ih1_hazy.png` paired with `clear/ih1.png`.
If a sibling `val/` directory exists, each epoch reports train/validation loss,
PSNR and SSIM. Use `--val-data PATH` to select another validation directory.

The model is intentionally usable without a checkpoint (the untrained network is useful for integration tests; train it on paired hazy/clear data for restoration quality). `haze_wavelet_loss` implements the paper's amplitude/decomposition objective.

## Colab throughput diagnostics

The Colab notebook exposes worker count, prefetching, a physical micro-batch
limit, and AMP settings in Cell 2. It keeps the existing paper batch sizes;
changing an effective batch from 16 to 512 changes training, not just performance.
If the configured batch is 512, the default micro-batch limit of 64 uses eight
gradient accumulation steps. Compare physical batches of 64, 128 and 256 using
measured throughput rather than trying to fill VRAM.

`--num-workers 4 --prefetch-factor 2 --pin-memory --persistent-workers` enables
parallel input loading. Reduce workers/prefetch if host RAM or shared memory is
exhausted. Prefetched paired float32 batches consume approximately
`workers * prefetch * micro_batch * 2 * 3 * size * size * 4` bytes, excluding
worker overhead and the active batch (about 768 MiB for 4/2/64 at size 256).
`--amp --amp-dtype bfloat16` enables BF16 on supported CUDA devices; the CLI
defaults to FP16 when `--amp-dtype` is omitted.

`--profile-batches 8` reports input wait, host-to-device transfer, and training
plus metrics time for the first eight batches of epoch 1. It synchronizes CUDA
for these measurements and includes startup/cuDNN warmup; set it to 0 after
diagnosis. Every epoch also prints images/s; CUDA profiling reports peak allocated
and reserved memory. Multi-scale training randomly selects 64, 128 or 256 pixels
per batch, so memory and compute requirements naturally vary. Set
`USE_MULTI_SCALE = False` only for a controlled fixed-resolution benchmark;
this is not the same training recipe.
