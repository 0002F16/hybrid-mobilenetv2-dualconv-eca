"""
MobileNetV2 Hybrid: lightweight DualConv + ECA in bottlenecks B4-B10.

This variant keeps the efficient DualConvBlock path and appends ECA directly
on the block output for B4..B10 replacements. Blocks outside B4..B10 use the
baseline InvertedResidual unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.dualconv import DualConvBlock
from models.eca import ECA
from models.mobilenetv2_baseline import (
    ConvBNReLU,
    InvertedResidual,
    REPLACEMENT_BLOCK_NAMES,
    _INVERTED_RESIDUAL_SETTING,
    _make_divisible,
)


class InvertedResidualDualConvECA(nn.Module):
    """
    Lightweight hybrid block: DualConvBlock -> ECA.

    `expand_ratio` is intentionally accepted to keep call signatures aligned
    with baseline/hybrid block construction, but it is not used in this
    lightweight path.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        expand_ratio: float,
        *,
        dualconv_groups: int = 4,
        eca_gamma: int = 2,
        eca_b: int = 1,
    ) -> None:
        super().__init__()
        _ = expand_ratio
        self.dualconv_block = DualConvBlock(
            in_ch,
            out_ch,
            stride=stride,
            groups=dualconv_groups,
        )
        self.eca = ECA(out_ch, gamma=eca_gamma, b=eca_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dualconv_block(x)
        return self.eca(out)


class MobileNetV2Hybrid(nn.Module):
    """
    MobileNetV2 with DualConv + ECA applied to bottlenecks B4–B10.

    Blocks outside the replacement scope remain baseline InvertedResidual.
    Shares the same stem, head, classifier, stride policy, and weight-init
    as MobileNetV2Baseline and the single-modification variants.
    """

    def __init__(
        self,
        num_classes: int,
        width_mult: float = 1.0,
        round_nearest: int = 8,
        dropout: float = 0.2,
        small_input: bool = False,
        *,
        dualconv_groups: int = 4,
        eca_gamma: int = 2,
        eca_b: int = 1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.width_mult = width_mult
        self.small_input = small_input
        self.dualconv_groups = dualconv_groups
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
                    self.blocks[name] = InvertedResidualDualConvECA(
                        current_ch,
                        out_ch,
                        stride,
                        float(t),
                        dualconv_groups=dualconv_groups,
                        eca_gamma=eca_gamma,
                        eca_b=eca_b,
                    )
                else:
                    self.blocks[name] = InvertedResidual(
                        current_ch, out_ch, stride, float(t)
                    )
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
            f"MobileNetV2Hybrid(num_classes={self.num_classes}, "
            f"width_mult={self.width_mult}, small_input={self.small_input}, "
            f"dualconv_groups={self.dualconv_groups}, "
            f"eca_gamma={self.eca_gamma}, eca_b={self.eca_b}, "
            f"replacement_scope={REPLACEMENT_BLOCK_NAMES})"
        )
