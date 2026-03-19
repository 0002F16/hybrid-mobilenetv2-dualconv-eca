from __future__ import annotations

from typing import Any

import torch.nn as nn

from models.hybrid import HybridMobileNetV2
from models.mobilenetv2_baseline import MobileNetV2Baseline
from models.mobilenetv2_dualconv_variants import (
    MobileNetV2DualConvAll,
    MobileNetV2DualConvB4B10,
    MobileNetV2DualConvB4B7,
)


def build_model(cfg: dict[str, Any]) -> nn.Module:
    """
    Build a model from config.

    Baseline-first: supports `baseline`, `dualconv_*`, and `hybrid`.
    The function is intentionally centralized so additional thesis variants can
    be added without touching training scripts.
    """
    model_name = str(cfg.get("model", "baseline")).lower()
    dataset = str(cfg.get("dataset", "")).lower()
    num_classes = int(cfg["num_classes"])
    width_mult = float(cfg.get("width_multiplier", 1.0))

    is_cifar = dataset in {"cifar10", "cifar100"}
    dualconv_groups = int(cfg.get("dualconv_groups", 4))

    if model_name in {"baseline", "mobilenetv2_baseline", "mobilenetv2"}:
        return MobileNetV2Baseline(
            num_classes=num_classes,
            width_mult=width_mult,
            small_input=is_cifar,
        )

    if model_name in {"dualconv_all", "mobilenetv2_dualconv_all"}:
        return MobileNetV2DualConvAll(
            num_classes=num_classes,
            width_mult=width_mult,
            small_input=is_cifar,
            dualconv_groups=dualconv_groups,
        )

    if model_name in {"dualconv_b4b10", "mobilenetv2_dualconv_b4b10"}:
        return MobileNetV2DualConvB4B10(
            num_classes=num_classes,
            width_mult=width_mult,
            small_input=is_cifar,
            dualconv_groups=dualconv_groups,
        )

    if model_name in {"dualconv_b4b7", "mobilenetv2_dualconv_b4b7"}:
        return MobileNetV2DualConvB4B7(
            num_classes=num_classes,
            width_mult=width_mult,
            small_input=is_cifar,
            dualconv_groups=dualconv_groups,
        )

    if model_name in {"hybrid", "hybrid_mobilenetv2"}:
        input_size = int(cfg.get("input_size", 32 if is_cifar else 64))
        return HybridMobileNetV2(
            num_classes=num_classes,
            width_multiplier=width_mult,
            input_size=input_size,
        )

    raise ValueError(
        f"Unknown model '{model_name}'. Supported: baseline, dualconv_all, "
        "dualconv_b4b10, dualconv_b4b7, hybrid."
    )

