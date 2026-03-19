from __future__ import annotations

from typing import Any

import torch.nn as nn

from models.hybrid import MobileNetV2Hybrid
from models.mobilenetv2_baseline import MobileNetV2Baseline
from models.mobilenetv2_dualconv_variants import (
    MobileNetV2DualConvAll,
    MobileNetV2DualConvB4B10,
    MobileNetV2DualConvB4B7,
)
from models.mobilenetv2_eca import MobileNetV2ECAOnly


def build_model(cfg: dict[str, Any]) -> nn.Module:
    """
    Build a model from config.

    Supports all four thesis variants plus DualConv sub-variants.
    The function is intentionally centralized so additional thesis variants can
    be added without touching training scripts.
    """
    model_name = str(cfg.get("model", "baseline")).lower()
    dataset = str(cfg.get("dataset", "")).lower()
    num_classes = int(cfg["num_classes"])
    width_mult = float(cfg.get("width_multiplier", 1.0))

    is_cifar = dataset in {"cifar10", "cifar100"}
    dualconv_groups = int(cfg.get("dualconv_groups", 4))
    eca_gamma = int(cfg.get("eca_gamma", 2))
    eca_b = int(cfg.get("eca_b", 1))

    if model_name in {"baseline", "mobilenetv2_baseline", "mobilenetv2"}:
        return MobileNetV2Baseline(
            num_classes=num_classes,
            width_mult=width_mult,
            small_input=is_cifar,
        )

    if model_name in {"dualconv", "dualconv_b4b10", "mobilenetv2_dualconv_b4b10"}:
        return MobileNetV2DualConvB4B10(
            num_classes=num_classes,
            width_mult=width_mult,
            small_input=is_cifar,
            dualconv_groups=dualconv_groups,
        )

    if model_name in {"dualconv_all", "mobilenetv2_dualconv_all"}:
        return MobileNetV2DualConvAll(
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

    if model_name in {"eca", "eca_only", "mobilenetv2_eca"}:
        return MobileNetV2ECAOnly(
            num_classes=num_classes,
            width_mult=width_mult,
            small_input=is_cifar,
            eca_gamma=eca_gamma,
            eca_b=eca_b,
        )

    if model_name in {"hybrid", "hybrid_mobilenetv2", "dualconv_eca"}:
        return MobileNetV2Hybrid(
            num_classes=num_classes,
            width_mult=width_mult,
            small_input=is_cifar,
            dualconv_groups=dualconv_groups,
            eca_gamma=eca_gamma,
            eca_b=eca_b,
        )

    raise ValueError(
        f"Unknown model '{model_name}'. Supported: baseline, dualconv, "
        "dualconv_all, dualconv_b4b10, dualconv_b4b7, eca, hybrid."
    )
