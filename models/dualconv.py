"""
DualConv modules (Zhong et al., 2022): 3x3 group conv + 1x1 pointwise conv on the same input.

DualConv2d computes:
    y = Conv3x3_group(x) + Conv1x1_pointwise(x)
and DualConvBlock applies BN+ReLU6 after DualConv, with an optional residual connection.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DualConv2d(nn.Module):
    """
    DualConv layer: 3x3 group convolution + 1x1 pointwise convolution in parallel, summed.

    PyTorch `groups` means both `in_channels` and `out_channels` must be divisible by `groups`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        groups: int = 4,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError(
                f"DualConv2d requires in/out channels divisible by groups. "
                f"Got in={in_channels}, out={out_channels}, groups={groups}."
            )

        self.conv3x3 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=groups,
            bias=bias,
        )
        self.conv1x1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            groups=1,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv3x3(x) + self.conv1x1(x)


class DualConvBlock(nn.Module):
    """
    Paper-style MobileNetV2 replacement block:
        DualConv -> BN -> ReLU6
    with residual connection only when stride == 1 and in_channels == out_channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
        groups: int = 4,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.use_res_connect = stride == 1 and in_channels == out_channels

        self.dualconv = DualConv2d(
            in_channels,
            out_channels,
            stride=stride,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn(self.dualconv(x)))
        if self.use_res_connect:
            return x + out
        return out

