"""A compact implementation of the HazeWaveNet architecture from the paper.

The fixed Haar analysis is deliberately kept separate from the learnable network:
LL is used for haze correction, while LH/HL/HH receive a lightweight detail path.
"""
from __future__ import annotations

import time
import torch
from torch import Tensor, nn
import torch.nn.functional as F


def haar_dwt(x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Differentiable 2-D Haar decomposition (symmetric padding for odd sizes)."""
    if x.shape[-2] % 2 or x.shape[-1] % 2:
        x = F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), mode="reflect")
    a, b = x[..., 0::2, :], x[..., 1::2, :]
    ll = (a[..., 0::2] + a[..., 1::2] + b[..., 0::2] + b[..., 1::2]) * 0.5
    lh = (a[..., 0::2] - a[..., 1::2] + b[..., 0::2] - b[..., 1::2]) * 0.5
    hl = (a[..., 0::2] + a[..., 1::2] - b[..., 0::2] - b[..., 1::2]) * 0.5
    hh = (a[..., 0::2] - a[..., 1::2] - b[..., 0::2] + b[..., 1::2]) * 0.5
    return ll, lh, hl, hh


def dark_trend(x: Tensor, patch: int = 15, omega: float = 0.95) -> tuple[Tensor, Tensor]:
    """DCP-inspired transmittance and normalized haze trend map."""
    dark = x.min(dim=1, keepdim=True).values
    # max-pooling on the negative image is a fast differentiable erosion.
    dark = -F.max_pool2d(-dark, patch, stride=1, padding=patch // 2)
    flat_dark = dark.flatten(1)
    top_count = max(1, int(flat_dark.shape[1] * 0.001))
    atmospheric = flat_dark.topk(top_count, dim=1).values.mean(dim=1, keepdim=True).view(-1, 1, 1, 1).clamp_min(1e-3)
    trans = (1.0 - omega * dark / atmospheric).clamp(0.1, 1.0)
    mean = trans.mean(dim=(2, 3), keepdim=True)
    std = trans.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(0.01)
    trend = torch.sigmoid((trans - mean) / std)
    return trans, trend


class GuidedBlock(nn.Module):
    def __init__(self, channels: int = 32):
        super().__init__()
        self.factor = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 3), padding=(0, 1), padding_mode="reflect"),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, (3, 1), padding=(1, 0), padding_mode="reflect"),
        )
        self.channel = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1), nn.Sigmoid())
        self.spatial = nn.Sequential(nn.Conv2d(1, 1, 7, padding=3, padding_mode="reflect"), nn.Sigmoid())
        self.guide = nn.Conv2d(1, channels, 1)
        self.norm = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x: Tensor, trend: Tensor) -> Tensor:
        t = F.interpolate(trend, size=x.shape[-2:], mode="bilinear", align_corners=False)
        guided = x * (1.0 + torch.sigmoid(self.guide(t)))
        y = self.factor(guided)
        y = y * self.channel(y) * self.spatial(y.mean(1, keepdim=True))
        return self.norm(x + y)


class GuidedGroup(nn.Module):
    def __init__(self, channels: int, blocks: int):
        super().__init__()
        self.blocks = nn.ModuleList([GuidedBlock(channels) for _ in range(blocks)])

    def forward(self, x: Tensor, trend: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x, trend)
        return x

class VariationPath(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.vp = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(channels),
        )
        self.vr = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        processed = self.vp(x)
        reduced = F.interpolate(self.vr(x), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return processed + reduced


class HazeWaveNet(nn.Module):
    """Parameter-efficient four-level Haar wavelet dehazer (paper's default)."""
    def __init__(self, channels: int = 32, levels: int = 4, blocks: int = 3):
        super().__init__()
        if not 1 <= levels <= 4:
            raise ValueError("levels must be between 1 and 4")
        self.levels = levels
        self.stem = nn.Conv2d(3, channels, 3, padding=1, padding_mode="reflect")
        self.low_projection = nn.ModuleList([nn.Conv2d(channels, channels, 1) for _ in range(levels)])
        self.low = nn.ModuleList([GuidedGroup(channels, blocks) for _ in range(levels)])
        self.high_projection = nn.ModuleList([nn.Conv2d(channels * 3, channels, 1) for _ in range(levels)])
        self.high = nn.ModuleList([VariationPath(channels) for _ in range(levels)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(channels, channels, 2, stride=2) for _ in range(levels - 1)])
        self.merge = nn.ModuleList([nn.Conv2d(channels * 2, channels, 3, padding=1) for _ in range(levels)])
        self.region = nn.ModuleList([nn.Sequential(nn.Conv2d(channels, 1, 1), nn.Sigmoid()) for _ in range(levels)])
        self.out = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, padding_mode="reflect"), nn.ReLU(inplace=True), nn.Conv2d(channels, 3, 3, padding=1, padding_mode="reflect"))

    def forward(self, x: Tensor) -> Tensor:
        original_size = x.shape[-2:]
        trans, trend = dark_trend(x)
        feat = self.stem(x)
        details: list[Tensor] = []
        for i in range(self.levels):
            ll, lh, hl, hh = haar_dwt(feat)
            feat = self.low[i](self.low_projection[i](ll), trend)
            high = self.high_projection[i](torch.cat((lh, hl, hh), dim=1))
            details.append(self.high[i](high))
        # WIM: learnable coarse-to-fine reconstruction, retaining every detail scale.
        for i in range(self.levels - 1, -1, -1):
            if i < self.levels - 1:
                feat = self.up[i](feat)
            if feat.shape[-2:] != details[i].shape[-2:]:
                feat = F.interpolate(feat, size=details[i].shape[-2:], mode="bilinear", align_corners=False)
            fused = F.relu(self.merge[i](torch.cat((feat, details[i]), dim=1)), inplace=True)
            feat = fused * self.region[i](fused) + feat
        result = torch.tanh(self.out(feat)).add(1.0).mul(0.5)
        result = F.interpolate(result, size=original_size, mode="bilinear", align_corners=False)
        return result.clamp(0.0, 1.0)

    @torch.inference_mode()
    def profile_forward(self, x: Tensor, repeats: int = 1) -> tuple[Tensor, dict[str, float]]:
        """Profile inference stages in milliseconds, synchronizing CUDA when needed."""
        if repeats < 1:
            raise ValueError("repeats must be positive")

        def now() -> float:
            if x.is_cuda:
                torch.cuda.synchronize(x.device)
            return time.perf_counter()

        totals = {name: 0.0 for name in (
            "dcp_fixed_prior", "dwt_mfde_decomposition", "low_frequency_fegg_fegb",
            "high_frequency_vp_vr", "wim_reconstruction", "final_refinement", "total",
        )}
        result = None
        for _ in range(repeats):
            start = now()
            t = now(); _, trend = dark_trend(x); totals["dcp_fixed_prior"] += now() - t
            t = now(); feat = self.stem(x); totals["dwt_mfde_decomposition"] += now() - t
            details = []
            for i in range(self.levels):
                t = now(); ll, lh, hl, hh = haar_dwt(feat); totals["dwt_mfde_decomposition"] += now() - t
                t = now(); feat = self.low[i](self.low_projection[i](ll), trend); totals["low_frequency_fegg_fegb"] += now() - t
                t = now(); high = self.high_projection[i](torch.cat((lh, hl, hh), dim=1)); details.append(self.high[i](high)); totals["high_frequency_vp_vr"] += now() - t
            t = now()
            for i in range(self.levels - 1, -1, -1):
                if i < self.levels - 1:
                    feat = self.up[i](feat)
                if feat.shape[-2:] != details[i].shape[-2:]:
                    feat = F.interpolate(feat, size=details[i].shape[-2:], mode="bilinear", align_corners=False)
                fused = F.relu(self.merge[i](torch.cat((feat, details[i]), dim=1)), inplace=True)
                feat = fused * self.region[i](fused) + feat
            totals["wim_reconstruction"] += now() - t
            t = now()
            result = torch.tanh(self.out(feat)).add(1.0).mul(0.5)
            result = F.interpolate(result, size=x.shape[-2:], mode="bilinear", align_corners=False)
            result = result.clamp(0.0, 1.0)
            totals["final_refinement"] += now() - t
            totals["total"] += now() - start
        return result, {name: value * 1000.0 / repeats for name, value in totals.items()}


def haze_wavelet_loss(pred: Tensor, target: Tensor, theta: float = 0.2) -> Tensor:
    """Paper Eq. (8-9): amplitude L1 plus four-band decomposition MSE."""
    p, t = pred, target
    p_bands, t_bands = [], []
    for _ in range(4):
        p, *pb = haar_dwt(p); t, *tb = haar_dwt(t)
        p_bands.extend([p, *pb]); t_bands.extend([t, *tb])
    decomposition = sum(F.mse_loss(a, b) for a, b in zip(p_bands, t_bands)) / len(p_bands)
    amplitude = sum(F.l1_loss(a.abs(), b.abs()) for a, b in zip(p_bands, t_bands)) / len(p_bands)
    return theta * amplitude + (1.0 - theta) * decomposition
