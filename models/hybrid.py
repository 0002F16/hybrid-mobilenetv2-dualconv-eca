"""
Hybrid MobileNetV2: backbone + efficient convolution + lightweight attention.

Combines MobileNetV2 backbone with attention modules for complex image classification.
"""

import torch
import torch.nn as nn

from models.attention import LightweightAttention
from models.efficient_conv import InvertedResidual


def _make_divisible(v: float, divisor: int, min_value: int | None = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class HybridMobileNetV2(nn.Module):
    """
    Hybrid MobileNetV2 with lightweight attention.

    Inverted residual blocks with optional SE-style attention after selected blocks.
    """

    def __init__(
        self,
        num_classes: int = 10,
        width_multiplier: float = 1.0,
        input_size: int = 32,
        round_nearest: int = 8,
        attention_reduction: int = 16,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size

        input_channel = 32
        last_channel = 1280
        input_channel = _make_divisible(input_channel * width_multiplier, round_nearest)
        self.last_channel = _make_divisible(
            last_channel * max(1.0, width_multiplier), round_nearest
        )

        self.features: list[nn.Module] = []
        self.features.extend([
            nn.Conv2d(3, input_channel, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True),
        ])

        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 1 if input_size <= 32 else 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # Add attention after blocks with sufficient channels (e.g. 64+)
        attention_positions = {4, 5, 6}  # After 64, 96, 160 channel blocks

        block_idx = 0
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_multiplier, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(
                    InvertedResidual(input_channel, output_channel, stride, t)
                )
                input_channel = output_channel
                if block_idx in attention_positions and output_channel >= 32:
                    self.features.append(
                        LightweightAttention(output_channel, attention_reduction)
                    )
                block_idx += 1

        self.features.append(
            nn.Conv2d(input_channel, self.last_channel, 1, bias=False),
        )
        self.features.append(nn.BatchNorm2d(self.last_channel))
        self.features.append(nn.ReLU6(inplace=True))

        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
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
