"""
MobileNetV2 backbone for image classification.

Provides the base feature extractor with inverted residual blocks.
Adapted for CIFAR (32x32) and Tiny-ImageNet (64x64) input sizes.
"""

import torch
import torch.nn as nn

from models.efficient_conv import InvertedResidual


def _make_divisible(v: float, divisor: int, min_value: int | None = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MobileNetV2Backbone(nn.Module):
    """
    MobileNetV2 backbone with configurable width and input size.

    Architecture uses inverted residual blocks with linear bottlenecks.
    """

    def __init__(
        self,
        num_classes: int = 10,
        width_multiplier: float = 1.0,
        input_size: int = 32,
        round_nearest: int = 8,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size

        # CIFAR/Tiny-ImageNet: first conv uses stride 1, smaller kernel
        input_channel = 32
        last_channel = 1280
        input_channel = _make_divisible(input_channel * width_multiplier, round_nearest)
        self.last_channel = _make_divisible(
            last_channel * max(1.0, width_multiplier), round_nearest
        )

        features: list[nn.Module] = [
            nn.Conv2d(3, input_channel, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True),
        ]

        # Inverted residual blocks (simplified for small inputs)
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1 if input_size <= 32 else 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_multiplier, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    InvertedResidual(input_channel, output_channel, stride, t)
                )
                input_channel = output_channel

        features.append(
            nn.Conv2d(input_channel, self.last_channel, 1, bias=False),
        )
        features.append(nn.BatchNorm2d(self.last_channel))
        features.append(nn.ReLU6(inplace=True))

        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )
        self._init_weights()

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

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
