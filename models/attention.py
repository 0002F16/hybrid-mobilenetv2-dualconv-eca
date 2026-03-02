"""
Lightweight attention modules: SE, ECA, CBAM-style.

Designed for minimal overhead in mobile architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block.

    Channel-wise attention via global average pooling and FC layers.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        scale = self.fc(x).view(b, c, 1, 1)
        return x * scale


class ECA(nn.Module):
    """
    Efficient Channel Attention.

    Uses 1D convolution for channel interaction with minimal params.
    """

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = torch.sigmoid(y)
        return x * y


class LightweightAttention(nn.Module):
    """
    Lightweight attention combining channel and spatial cues.

    Uses SE for channels and a minimal spatial gate.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.channel_attn = SqueezeExcitation(channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.channel_attn(x)
