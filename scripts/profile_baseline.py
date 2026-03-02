"""
Profile MobileNetV2 baseline: params, FLOPs/MACs, model size, latency.

Run from project root:
  python scripts/profile_baseline.py --input_res 32
  python scripts/profile_baseline.py --input_res 64 --num_classes 200

FLOPs profiling uses fvcore (thesis) with thop fallback. See README: latency
is intended to be measured on local hardware; other metrics can be run on Colab.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Run from project root; add project root to path when running script directly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch

from models.mobilenetv2_baseline import MobileNetV2Baseline
from utils.profiling import (
    count_parameters,
    compute_flops,
    measure_latency,
    measure_model_size_mb,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile MobileNetV2 baseline")
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
        help="Width multiplier for baseline",
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

    model = MobileNetV2Baseline(
        num_classes=num_classes,
        width_mult=args.width_mult,
        dropout=0.2,
        small_input=args.small_input,
    )

    nparams = count_parameters(model)
    flops_result = compute_flops(model, input_size, device)
    size_mb = measure_model_size_mb(model)
    latency_ms = measure_latency(
        model, input_size, device, warmup=30, iters=200, batch_size=args.batch_size
    )

    lines = [
        f"Model: MobileNetV2 Baseline (width_mult={args.width_mult})",
        f"Params: {nparams} ({nparams / 1e6:.2f} M)",
        f"Input: 1x{input_size[0]}x{input_size[1]}x{input_size[2]}",
        f"MACs: {flops_result['macs']}",
        f"FLOPs: {flops_result['flops']} (method={flops_result['method_used']})",
        f"Model size: {size_mb:.4f} MB",
        f"Latency: {latency_ms:.4f} ms/image (device={device})",
    ]
    report = "\n".join(lines)

    print(report)
    if args.output:
        Path(args.output).write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
