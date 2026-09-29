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
