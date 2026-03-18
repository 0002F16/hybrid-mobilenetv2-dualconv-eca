"""
ECA-Net (Wang et al., 2020): Efficient Channel Attention (paper-faithful).

- GAP (no dimensionality reduction)
- 1D convolution over channels with adaptive kernel size k (Eq. 12)
- sigmoid gating and channel-wise reweighting
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def eca_kernel_size(channels: int, gamma: int = 2, b: int = 1) -> int:
    """
    Adaptive kernel size selection from ECA-Net (Eq. 12).

    k = | (log2(C) + b) / gamma |_odd
    where |_odd is the nearest odd integer.
    """
    if channels <= 0:
        raise ValueError(f"channels must be > 0, got {channels}")
    t = (math.log2(float(channels)) + float(b)) / float(gamma)
    k = int(round(t))
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1
    return k


class ECA(nn.Module):
    """
    Efficient Channel Attention with adaptive kernel size (paper default).

    Input:  (B, C, H, W)
    Output: (B, C, H, W)
    """

    def __init__(self, channels: int, gamma: int = 2, b: int = 1) -> None:
        super().__init__()
        k = eca_kernel_size(channels, gamma=gamma, b=b)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)  # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)  # (B, 1, C)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        y = torch.sigmoid(y)
        return x * y

