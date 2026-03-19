"""
Profile all four thesis variants: params, FLOPs/MACs, model size, latency.

Phase 6 deliverable — all-variant verification table.
Builds each variant via build_model, verifies forward-pass output shapes,
and flags any variant exceeding the ±10% efficiency budget vs baseline.

Run from project root:
  python scripts/profile_all_variants.py
  python scripts/profile_all_variants.py --input_res 64 --num_classes 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch

from models.factory import build_model
from utils.profiling import (
    count_parameters,
    compute_flops,
    measure_latency,
    measure_model_size_mb,
)

THESIS_VARIANTS = ["baseline", "dualconv", "eca", "hybrid"]
BUDGET_TOLERANCE_PCT = 10.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile all four thesis MobileNetV2 variants"
    )
    parser.add_argument(
        "--num_classes", type=int, default=None,
        help="Number of classes (default: 10 for 32x32, 200 for 64x64)",
    )
    parser.add_argument(
        "--input_res", type=int, default=32, choices=[32, 64],
        help="Spatial resolution H=W (32 for CIFAR, 64 for Tiny-ImageNet)",
    )
    parser.add_argument(
        "--width_mult", type=float, default=1.0,
        help="Width multiplier",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size for latency measurement",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Write JSON results to this file",
    )
    parser.add_argument(
        "--skip_latency", action="store_true",
        help="Skip latency measurement (faster for CI)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    num_classes = args.num_classes
    if num_classes is None:
        num_classes = 10 if args.input_res == 32 else 200

    is_cifar = args.input_res == 32
    dataset = "cifar10" if is_cifar else "tiny_imagenet"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = (3, args.input_res, args.input_res)

    results: list[dict] = []
    baseline_params = 0
    baseline_flops = 0

    header = (
        f"{'Variant':<12} {'Params':>12} {'Params%':>9} "
        f"{'FLOPs':>14} {'FLOPs%':>9} {'Size(MB)':>10} {'Latency(ms)':>12} {'Budget'}"
    )
    print(f"\nAll-Variant Verification Table  (input={args.input_res}x{args.input_res}, classes={num_classes})")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for variant in THESIS_VARIANTS:
        cfg = {
            "model": variant,
            "dataset": dataset,
            "num_classes": num_classes,
            "width_multiplier": args.width_mult,
        }
        model = build_model(cfg)

        # Forward-pass shape verification
        dummy = torch.zeros(1, *input_size)
        model.eval()
        with torch.no_grad():
            out = model(dummy)
        expected_shape = (1, num_classes)
        assert out.shape == expected_shape, (
            f"{variant}: expected output {expected_shape}, got {out.shape}"
        )

        nparams = count_parameters(model)
        flops_result = compute_flops(model, input_size, device)
        flops = flops_result["flops"]
        size_mb = measure_model_size_mb(model)

        latency_ms = -1.0
        if not args.skip_latency:
            latency_ms = measure_latency(
                model, input_size, device,
                warmup=30, iters=200, batch_size=args.batch_size,
            )

        if variant == "baseline":
            baseline_params = nparams
            baseline_flops = flops
            params_pct = 0.0
            flops_pct = 0.0
        else:
            params_pct = (nparams - baseline_params) / baseline_params * 100.0 if baseline_params else 0.0
            flops_pct = (flops - baseline_flops) / baseline_flops * 100.0 if baseline_flops else 0.0

        within_budget = abs(params_pct) <= BUDGET_TOLERANCE_PCT and abs(flops_pct) <= BUDGET_TOLERANCE_PCT
        budget_flag = "OK" if within_budget else "OVER"

        latency_str = f"{latency_ms:>10.4f}ms" if latency_ms >= 0 else f"{'skipped':>12}"
        print(
            f"{variant:<12} {nparams:>12,} {params_pct:>+8.2f}% "
            f"{flops:>14,} {flops_pct:>+8.2f}% {size_mb:>9.4f}MB {latency_str} {budget_flag}"
        )

        results.append({
            "variant": variant,
            "params": nparams,
            "params_pct_vs_baseline": round(params_pct, 4),
            "flops": flops,
            "flops_pct_vs_baseline": round(flops_pct, 4),
            "macs": flops_result["macs"],
            "flops_method": flops_result["method_used"],
            "size_mb": round(size_mb, 6),
            "latency_ms": round(latency_ms, 6) if latency_ms >= 0 else None,
            "within_budget": within_budget,
            "input_res": args.input_res,
            "num_classes": num_classes,
        })

    print("=" * len(header))
    print(f"Budget tolerance: ±{BUDGET_TOLERANCE_PCT}% of baseline params/FLOPs\n")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
