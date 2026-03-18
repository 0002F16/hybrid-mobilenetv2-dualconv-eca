"""
Profile MobileNetV2 baseline vs MobileNetV2-ECA (no DualConv):
params, FLOPs/MACs, model size, latency, and % deltas vs baseline.

Run from project root:
  python scripts/profile_eca_variants.py --input_res 32 --small_input
  python scripts/profile_eca_variants.py --input_res 64 --num_classes 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch

from models.mobilenetv2_baseline import MobileNetV2Baseline
from models.mobilenetv2_eca import MobileNetV2ECAOnly
from utils.profiling import (
    count_parameters,
    compute_flops,
    measure_latency,
    measure_model_size_mb,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile MobileNetV2 vs MobileNetV2-ECA")
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Number of classes (default: 10 for 32x32, 200 for 64x64)",
    )
    parser.add_argument(
        "--input_res",
        type=int,
        default=32,
        choices=[32, 64],
        help="Spatial resolution H=W (32 for CIFAR, 64 for Tiny-ImageNet)",
    )
    parser.add_argument(
        "--width_mult",
        type=float,
        default=1.0,
        help="Width multiplier for models",
    )
    parser.add_argument(
        "--small_input",
        action="store_true",
        help="Use stem stride 1 (CIFAR-style)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for latency measurement",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write report to this file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility",
    )
    return parser.parse_args()


def _pct_change(value: float, base: float) -> float:
    if base == 0:
        return 0.0
    return (value - base) / base * 100.0


def _profile_one(
    *,
    name: str,
    model: torch.nn.Module,
    input_size: tuple[int, int, int],
    device: torch.device,
    batch_size: int,
    baseline_params: int,
    baseline_flops: int,
    baseline_size_mb: float,
) -> str:
    nparams = count_parameters(model)
    flops_result = compute_flops(model, input_size, device)
    size_mb = measure_model_size_mb(model)
    latency_ms = measure_latency(
        model, input_size, device, warmup=30, iters=200, batch_size=batch_size
    )

    params_pct = _pct_change(float(nparams), float(baseline_params))
    flops_pct = _pct_change(float(flops_result["flops"]), float(baseline_flops))
    size_pct = _pct_change(float(size_mb), float(baseline_size_mb))

    lines = [
        f"Model: {name}",
        f"Params: {nparams} ({nparams / 1e6:.2f} M) ({params_pct:+.2f}% vs baseline)",
        f"Input: 1x{input_size[0]}x{input_size[1]}x{input_size[2]}",
        f"MACs: {flops_result['macs']}",
        f"FLOPs: {flops_result['flops']} (method={flops_result['method_used']}) ({flops_pct:+.2f}% vs baseline)",
        f"Model size: {size_mb:.4f} MB ({size_pct:+.2f}% vs baseline)",
        f"Latency: {latency_ms:.4f} ms/image (device={device})",
    ]
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    num_classes = args.num_classes
    if num_classes is None:
        num_classes = 10 if args.input_res == 32 else 200

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = (3, args.input_res, args.input_res)

    baseline_model = MobileNetV2Baseline(
        num_classes=num_classes,
        width_mult=args.width_mult,
        dropout=0.2,
        small_input=args.small_input,
    )
    eca_model = MobileNetV2ECAOnly(
        num_classes=num_classes,
        width_mult=args.width_mult,
        dropout=0.2,
        small_input=args.small_input,
        eca_gamma=2,
        eca_b=1,
    )

    baseline_params = count_parameters(baseline_model)
    baseline_flops = compute_flops(baseline_model, input_size, device)["flops"]
    baseline_size_mb = measure_model_size_mb(baseline_model)

    reports = [
        _profile_one(
            name=f"MobileNetV2 Baseline (width_mult={args.width_mult})",
            model=baseline_model,
            input_size=input_size,
            device=device,
            batch_size=args.batch_size,
            baseline_params=baseline_params,
            baseline_flops=baseline_flops,
            baseline_size_mb=baseline_size_mb,
        ),
        "-" * 60,
        _profile_one(
            name=f"MobileNetV2-ECA (B4..B10 only, adaptive k, width_mult={args.width_mult})",
            model=eca_model,
            input_size=input_size,
            device=device,
            batch_size=args.batch_size,
            baseline_params=baseline_params,
            baseline_flops=baseline_flops,
            baseline_size_mb=baseline_size_mb,
        ),
        "-" * 60,
    ]

    report = "\n".join(reports).rstrip() + "\n"
    print(report, end="")
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()

