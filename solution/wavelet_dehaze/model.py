"""A compact implementation of the HazeWaveNet architecture from the paper.

The fixed Haar analysis is deliberately kept separate from the learnable network:
LL is used for haze correction, while LH/HL/HH receive a lightweight detail path.
"""
from __future__ import annotations

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
    atmospheric = dark.flatten(1).amax(dim=1, keepdim=True).view(-1, 1, 1, 1).clamp_min(1e-3)
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


class HazeWaveNet(nn.Module):
    """Parameter-efficient four-level Haar wavelet dehazer (paper's default)."""
    def __init__(self, channels: int = 32, levels: int = 4, blocks: int = 3):
        super().__init__()
        if not 1 <= levels <= 4:
            raise ValueError("levels must be between 1 and 4")
        self.levels = levels
        self.stem = nn.Conv2d(3, channels, 3, padding=1, padding_mode="reflect")
        self.low = nn.ModuleList([GuidedGroup(channels, blocks) for _ in range(levels)])
        self.high = nn.ModuleList([nn.Sequential(nn.Conv2d(channels * 3, channels, 3, padding=1), nn.ReLU(inplace=True), nn.InstanceNorm2d(channels)) for _ in range(levels)])
        self.merge = nn.ModuleList([nn.Conv2d(channels * 2, channels, 3, padding=1) for _ in range(levels)])
        self.out = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, padding_mode="reflect"), nn.ReLU(inplace=True), nn.Conv2d(channels, 3, 3, padding=1, padding_mode="reflect"))

    def forward(self, x: Tensor) -> Tensor:
        original_size = x.shape[-2:]
        trans, trend = dark_trend(x)
        feat = self.stem(x)
        details: list[Tensor] = []
        for i in range(self.levels):
            ll, lh, hl, hh = haar_dwt(feat)
            feat = self.low[i](ll, trend)
            details.append(self.high[i](torch.cat((lh, hl, hh), dim=1)))
        # WIM: learnable coarse-to-fine reconstruction, retaining every detail scale.
        for i in range(self.levels - 1, -1, -1):
            feat = F.interpolate(feat, size=details[i].shape[-2:], mode="bilinear", align_corners=False)
            feat = F.relu(self.merge[i](torch.cat((feat, details[i]), dim=1)), inplace=True)
        result = torch.tanh(self.out(feat))
        result = F.interpolate(result, size=original_size, mode="bilinear", align_corners=False)
        return (result + x).clamp(0.0, 1.0)


def haze_wavelet_loss(pred: Tensor, target: Tensor, theta: float = 0.2) -> Tensor:
    """Paper Eq. (8-9): amplitude L1 plus four-band decomposition MSE."""
    p, t = pred, target
    p_bands, t_bands = [], []
    for _ in range(4):
        p, *pb = haar_dwt(p); t, *tb = haar_dwt(t)
        p_bands.extend([p, *pb]); t_bands.extend([t, *tb])
    decomposition = sum(F.mse_loss(a, b) for a, b in zip(p_bands, t_bands)) / len(p_bands)
    amplitude = F.l1_loss(pred, target)
    return theta * amplitude + (1.0 - theta) * decomposition
