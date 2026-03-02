"""
MobileNetV2 baseline with explicit B1–B17 inverted residual blocks.

Thesis-grade implementation from scratch (no torchvision model call).
Supports CIFAR-style (small_input=True) and ImageNet-style stems.
Modular for later DualConv/ECA replacements in B4–B10.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Thesis 3.7.2: modifications restricted to stages B4 through B10.
REPLACEMENT_BLOCK_NAMES = ("B4", "B5", "B6", "B7", "B8", "B9", "B10")

# Stage plan: (expansion t, output channels c, num blocks n, stride s)
# B1; B2-B3; B4-B6; B7-B10; B11-B13; B14-B16; B17
_INVERTED_RESIDUAL_SETTING = [
    (1, 16, 1, 1),   # B1
    (6, 24, 2, 2),   # B2, B3
    (6, 32, 3, 2),   # B4, B5, B6
    (6, 64, 4, 2),   # B7, B8, B9, B10
    (6, 96, 3, 1),   # B11, B12, B13
    (6, 160, 3, 2),  # B14, B15, B16
    (6, 320, 1, 1),  # B17
]


def _make_divisible(v: float, divisor: int, min_value: int | None = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Module):
    """Conv2d + BatchNorm2d + ReLU6. No bias on conv."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class InvertedResidual(nn.Module):
    """
    Inverted residual block: expansion -> depthwise -> linear bottleneck.

    - Expansion: 1x1 conv + BN + ReLU6 (skipped when t==1).
    - Depthwise: 3x3 depthwise conv + BN + ReLU6.
    - Projection: 1x1 conv + BN only (no activation; linear bottleneck).
    Residual connection only when stride == 1 and in_channels == out_channels.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        expand_ratio: float,
    ) -> None:
        super().__init__()
        self.stride = stride
        hidden_ch = int(round(in_ch * expand_ratio))
        # Residual only when stride==1 and same channels (preserves shape).
        self.use_res_connect = stride == 1 and in_ch == out_ch

        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_ch, hidden_ch, 1, bias=False),
                nn.BatchNorm2d(hidden_ch),
                nn.ReLU6(inplace=True),
            ])
        layers.extend([
            nn.Conv2d(
                hidden_ch, hidden_ch, 3, stride=stride, padding=1, groups=hidden_ch, bias=False
            ),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            # No activation here: linear bottleneck (design choice per MobileNetV2).
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2Baseline(nn.Module):
    """
    MobileNetV2 baseline with explicitly named blocks B1–B17.

    Stem: 3x3 conv stride 2 (or 1 if small_input). Head: 1x1 to 1280, GAP, classifier.
    """

    def __init__(
        self,
        num_classes: int,
        width_mult: float = 1.0,
        round_nearest: int = 8,
        dropout: float = 0.2,
        small_input: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.width_mult = width_mult
        self.small_input = small_input

        input_ch = _make_divisible(32 * width_mult, round_nearest)
        self.last_ch = _make_divisible(1280 * max(1.0, width_mult), round_nearest)

        # Stem: 3x3, stride 2 (ImageNet-style) or 1 (CIFAR-style)
        stem_stride = 1 if small_input else 2
        self.stem = ConvBNReLU(3, input_ch, kernel_size=3, stride=stem_stride)

        # Build B1–B17 from stage plan
        self.blocks = nn.ModuleDict()
        block_names = self._block_names_from_setting()
        current_ch = input_ch
        idx = 0
        for t, c, n, s in _INVERTED_RESIDUAL_SETTING:
            out_ch = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                name = block_names[idx]
                self.blocks[name] = InvertedResidual(current_ch, out_ch, stride, float(t))
                current_ch = out_ch
                idx += 1

        self.block_order = block_names

        # Head: 1x1 to 1280, BN, ReLU6
        self.head_conv = nn.Conv2d(current_ch, self.last_ch, 1, bias=False)
        self.head_bn = nn.BatchNorm2d(self.last_ch)
        self.head_act = nn.ReLU6(inplace=True)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.last_ch, num_classes),
        )
        self._init_weights()

    @staticmethod
    def _block_names_from_setting() -> list[str]:
        names: list[str] = []
        for _, _, n, _ in _INVERTED_RESIDUAL_SETTING:
            for _ in range(n):
                names.append(f"B{len(names) + 1}")
        return names

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for name in self.block_order:
            x = self.blocks[name](x)
        x = self.head_act(self.head_bn(self.head_conv(x)))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def __repr__(self) -> str:
        return (
            f"MobileNetV2Baseline(num_classes={self.num_classes}, "
            f"width_mult={self.width_mult}, small_input={self.small_input}, "
            f"blocks=B1..B17, replacement_scope={REPLACEMENT_BLOCK_NAMES})"
        )
