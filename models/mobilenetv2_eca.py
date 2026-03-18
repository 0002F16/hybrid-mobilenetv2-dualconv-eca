"""
MobileNetV2 with Efficient Channel Attention (ECA) only (no DualConv).

Follows ECA-Net (Wang et al., 2020) placement for MobileNetV2:
integrate attention before the residual connection in each bottleneck.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.eca import ECA
from models.mobilenetv2_baseline import (
    ConvBNReLU,
    InvertedResidual,
    REPLACEMENT_BLOCK_NAMES,
    _INVERTED_RESIDUAL_SETTING,
    _make_divisible,
)


class InvertedResidualECA(nn.Module):
    """
    MobileNetV2 inverted residual block with ECA applied before residual add.

    Baseline structure:
      (optional) expansion 1x1 -> depthwise 3x3 -> projection 1x1 (linear bottleneck)

    ECA placement:
      apply ECA after the projection BN (i.e. on block output), then add residual if enabled.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        expand_ratio: float,
        *,
        eca_gamma: int = 2,
        eca_b: int = 1,
    ) -> None:
        super().__init__()
        self.stride = stride
        hidden_ch = int(round(in_ch * expand_ratio))
        self.use_res_connect = stride == 1 and in_ch == out_ch

        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.extend(
                [
                    nn.Conv2d(in_ch, hidden_ch, 1, bias=False),
                    nn.BatchNorm2d(hidden_ch),
                    nn.ReLU6(inplace=True),
                ]
            )
        layers.extend(
            [
                nn.Conv2d(
                    hidden_ch,
                    hidden_ch,
                    3,
                    stride=stride,
                    padding=1,
                    groups=hidden_ch,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_ch),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.eca = ECA(out_ch, gamma=eca_gamma, b=eca_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.eca(out)
        if self.use_res_connect:
            return x + out
        return out


class MobileNetV2ECAOnly(nn.Module):
    """MobileNetV2 baseline topology with ECA applied to blocks B4..B10 only."""

    def __init__(
        self,
        num_classes: int,
        width_mult: float = 1.0,
        round_nearest: int = 8,
        dropout: float = 0.2,
        small_input: bool = False,
        *,
        eca_gamma: int = 2,
        eca_b: int = 1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.width_mult = width_mult
        self.small_input = small_input
        self.eca_gamma = eca_gamma
        self.eca_b = eca_b

        input_ch = _make_divisible(32 * width_mult, round_nearest)
        self.last_ch = _make_divisible(1280 * max(1.0, width_mult), round_nearest)

        stem_stride = 1 if small_input else 2
        self.stem = ConvBNReLU(3, input_ch, kernel_size=3, stride=stem_stride)

        self.blocks = nn.ModuleDict()
        block_names = self._block_names_from_setting()
        current_ch = input_ch
        idx = 0
        for t, c, n, s in _INVERTED_RESIDUAL_SETTING:
            out_ch = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                name = block_names[idx]
                if name in REPLACEMENT_BLOCK_NAMES:
                    self.blocks[name] = InvertedResidualECA(
                        current_ch,
                        out_ch,
                        stride,
                        float(t),
                        eca_gamma=eca_gamma,
                        eca_b=eca_b,
                    )
                else:
                    self.blocks[name] = InvertedResidual(current_ch, out_ch, stride, float(t))
                current_ch = out_ch
                idx += 1

        self.block_order = block_names

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
            f"MobileNetV2ECAOnly(num_classes={self.num_classes}, width_mult={self.width_mult}, "
            f"small_input={self.small_input}, eca_gamma={self.eca_gamma}, eca_b={self.eca_b}, "
            f"replacement_scope={REPLACEMENT_BLOCK_NAMES})"
        )

