"""
Phase 6 exit-criteria tests: all four thesis variants build, forward-pass,
and use the correct block types in the replacement scope.
"""

from __future__ import annotations

import pytest
import torch

from models.factory import build_model
from models.hybrid import InvertedResidualDualConvECA, MobileNetV2Hybrid
from models.mobilenetv2_baseline import (
    InvertedResidual,
    MobileNetV2Baseline,
    REPLACEMENT_BLOCK_NAMES,
)
from models.mobilenetv2_dualconv_variants import MobileNetV2DualConvB4B10
from models.mobilenetv2_eca import InvertedResidualECA, MobileNetV2ECAOnly
from models.dualconv import DualConvBlock


THESIS_VARIANTS = ["baseline", "dualconv", "eca", "hybrid"]


def _make_cfg(model: str, *, input_res: int = 32, num_classes: int = 10) -> dict:
    dataset = "cifar10" if input_res == 32 else "tiny_imagenet"
    return {
        "model": model,
        "dataset": dataset,
        "num_classes": num_classes,
        "width_multiplier": 1.0,
    }


# ── Factory dispatch ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("variant", THESIS_VARIANTS)
def test_factory_builds_all_variants(variant: str) -> None:
    model = build_model(_make_cfg(variant))
    assert isinstance(model, torch.nn.Module)


def test_factory_baseline_type() -> None:
    assert isinstance(build_model(_make_cfg("baseline")), MobileNetV2Baseline)


def test_factory_dualconv_type() -> None:
    assert isinstance(build_model(_make_cfg("dualconv")), MobileNetV2DualConvB4B10)


def test_factory_eca_type() -> None:
    assert isinstance(build_model(_make_cfg("eca")), MobileNetV2ECAOnly)


def test_factory_hybrid_type() -> None:
    assert isinstance(build_model(_make_cfg("hybrid")), MobileNetV2Hybrid)


# ── Forward-pass shape verification ─────────────────────────────────────────

@pytest.mark.parametrize("variant", THESIS_VARIANTS)
@pytest.mark.parametrize(
    "input_res,num_classes",
    [(32, 10), (32, 100), (64, 200)],
    ids=["cifar10", "cifar100", "tinyimagenet"],
)
def test_forward_output_shape(variant: str, input_res: int, num_classes: int) -> None:
    cfg = _make_cfg(variant, input_res=input_res, num_classes=num_classes)
    model = build_model(cfg)
    model.eval()
    x = torch.randn(2, 3, input_res, input_res)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, num_classes), f"{variant}: {out.shape}"


# ── Block-type verification ─────────────────────────────────────────────────

def test_baseline_uses_only_inverted_residuals() -> None:
    model = build_model(_make_cfg("baseline"))
    assert isinstance(model, MobileNetV2Baseline)
    for name in model.block_order:
        assert isinstance(model.blocks[name], InvertedResidual), (
            f"Baseline block {name} should be InvertedResidual"
        )


def test_dualconv_b4b10_block_types() -> None:
    model = build_model(_make_cfg("dualconv"))
    assert isinstance(model, MobileNetV2DualConvB4B10)
    for name in model.block_order:
        block = model.blocks[name]
        if name in REPLACEMENT_BLOCK_NAMES:
            assert isinstance(block, DualConvBlock), (
                f"DualConv block {name} should be DualConvBlock"
            )
        else:
            assert isinstance(block, InvertedResidual), (
                f"DualConv block {name} outside scope should be InvertedResidual"
            )


def test_eca_only_block_types() -> None:
    model = build_model(_make_cfg("eca"))
    assert isinstance(model, MobileNetV2ECAOnly)
    for name in model.block_order:
        block = model.blocks[name]
        if name in REPLACEMENT_BLOCK_NAMES:
            assert isinstance(block, InvertedResidualECA), (
                f"ECA block {name} should be InvertedResidualECA"
            )
        else:
            assert isinstance(block, InvertedResidual), (
                f"ECA block {name} outside scope should be InvertedResidual"
            )


def test_hybrid_block_types() -> None:
    model = build_model(_make_cfg("hybrid"))
    assert isinstance(model, MobileNetV2Hybrid)
    for name in model.block_order:
        block = model.blocks[name]
        if name in REPLACEMENT_BLOCK_NAMES:
            assert isinstance(block, InvertedResidualDualConvECA), (
                f"Hybrid block {name} should be InvertedResidualDualConvECA"
            )
        else:
            assert isinstance(block, InvertedResidual), (
                f"Hybrid block {name} outside scope should be InvertedResidual"
            )


# ── Hybrid internals ────────────────────────────────────────────────────────

def test_hybrid_block_has_dualconv_and_eca() -> None:
    """Each hybrid replacement block must contain both DualConv2d and ECA."""
    from models.dualconv import DualConv2d
    from models.eca import ECA

    model = build_model(_make_cfg("hybrid"))
    assert isinstance(model, MobileNetV2Hybrid)
    for name in REPLACEMENT_BLOCK_NAMES:
        block = model.blocks[name]
        has_dualconv = any(isinstance(m, DualConv2d) for m in block.modules())
        has_eca = any(isinstance(m, ECA) for m in block.modules())
        assert has_dualconv, f"Hybrid block {name} missing DualConv2d"
        assert has_eca, f"Hybrid block {name} missing ECA"


# ── Replacement scope consistency ────────────────────────────────────────────

def test_replacement_scope_is_b4_through_b10() -> None:
    expected = ("B4", "B5", "B6", "B7", "B8", "B9", "B10")
    assert REPLACEMENT_BLOCK_NAMES == expected


def test_all_variants_have_17_blocks() -> None:
    for variant in THESIS_VARIANTS:
        model = build_model(_make_cfg(variant))
        assert len(model.block_order) == 17, (
            f"{variant} has {len(model.block_order)} blocks, expected 17"
        )
