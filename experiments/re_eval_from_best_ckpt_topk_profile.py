"""
Checkpoint re-evaluation for CIFAR-100 (Top-1 + Top-5) plus variant profiling.

For each trained run directory:
  OUTPUT_ROOT/cifar100/<variant>/seed_<seed>/
it loads `checkpoints/best.pt`, computes:
  - test.top1_pp
  - test.top5_pp
and writes them back to the existing `metrics.json` (in-place).

It also computes params/FLOPs/model size/latency once per variant and writes:
  OUTPUT_ROOT/cifar100/re_eval_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.preprocessing import get_dataset_loaders, get_transforms
from models.factory import build_model
from training.evaluate import evaluate_top1_top5
from training.utils import load_config
from utils.profiling import (
    count_parameters,
    compute_flops,
    measure_latency,
    measure_model_size_mb,
)


DEFAULT_SEEDS = [42, 123, 3407, 2024, 777]
DEFAULT_VARIANTS = ["baseline", "dualconv", "eca", "hybrid"]


def _run_dir(output_root: Path, *, dataset: str, variant: str, seed: int) -> Path:
    return output_root / dataset.lower() / variant.lower() / f"seed_{int(seed)}"


def _load_split_mean_std(artifacts_dir: Path, *, dataset: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    split_md_path = artifacts_dir / "split_metadata" / f"{dataset.lower()}.json"
    if not split_md_path.exists():
        raise FileNotFoundError(str(split_md_path))
    payload = json.loads(split_md_path.read_text(encoding="utf-8"))
    mean = tuple(float(x) for x in payload["mean"])
    std = tuple(float(x) for x in payload["std"])
    return mean, std


def _build_test_loader_from_split_md(
    *,
    cfg: dict[str, Any],
    dataset_root: str | Path,
    seed: int,
    artifacts_dir: Path,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    mean, std = _load_split_mean_std(artifacts_dir, dataset=str(cfg["dataset"]).lower())
    _train_t, test_t = get_transforms(str(cfg["dataset"]).lower(), mean=mean, std=std)
    test_dataset = CIFAR100(
        root=str(dataset_root),
        train=False,
        download=True,
        transform=test_t,
    )
    return DataLoader(
        test_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
    )


def _build_test_loader(
    *,
    cfg: dict[str, Any],
    dataset_root: str | Path,
    seed: int,
    artifacts_dir: Path,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    """
    Prefer to reuse existing split metadata (mean/std) if present, to avoid
    recomputing stats and splitting.
    """
    try:
        return _build_test_loader_from_split_md(
            cfg=cfg,
            dataset_root=dataset_root,
            seed=seed,
            artifacts_dir=artifacts_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    except FileNotFoundError:
        _train_loader, _val_loader, test_loader = get_dataset_loaders(
            dataset=str(cfg["dataset"]).lower(),
            root=str(dataset_root),
            batch_size=int(batch_size),
            num_workers=int(num_workers),
            pin_memory=bool(pin_memory),
            seed=int(seed),
            split_seed=int(cfg.get("split_seed", 1337)),
            artifacts_root=artifacts_dir,
        )
        return test_loader


def _profile_variant(
    *,
    variant: str,
    cfg: dict[str, Any],
    device: torch.device,
    input_res: int,
    num_classes: int,
    batch_size_latency: int,
    skip_latency: bool,
) -> dict[str, Any]:
    input_size = (3, input_res, input_res)
    profile_cfg = dict(cfg)
    profile_cfg["model"] = variant
    model = build_model(profile_cfg)

    nparams = count_parameters(model)
    flops_result = compute_flops(model, input_size, device)
    size_mb = measure_model_size_mb(model)
    latency_ms = -1.0
    if not skip_latency:
        latency_ms = measure_latency(
            model,
            input_size,
            device,
            warmup=30,
            iters=200,
            batch_size=int(batch_size_latency),
        )

    return {
        "variant": variant,
        "params": int(nparams),
        "flops": int(flops_result["flops"]),
        "macs": int(flops_result["macs"]),
        "flops_method": flops_result["method_used"],
        "size_mb": float(round(size_mb, 6)),
        "latency_ms": float(round(latency_ms, 6)) if latency_ms >= 0 else None,
        "input_res": int(input_res),
        "num_classes": int(num_classes),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output_root", default="outputs", help="Root output directory used by training.")
    p.add_argument("--dataset_root", default="./data", help="Dataset root (e.g. /content/data in Colab).")
    p.add_argument("--dataset", default="cifar100", choices=["cifar10", "cifar100", "tiny_imagenet"])
    p.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=DEFAULT_SEEDS,
        help="Seeds to re-evaluate (default: thesis seeds).",
    )
    p.add_argument(
        "--variants",
        nargs="*",
        default=DEFAULT_VARIANTS,
        help="Model variants to re-evaluate (baseline/dualconv/eca/hybrid).",
    )
    p.add_argument("--config", default="configs/cifar100.yaml", help="Template config YAML.")
    p.add_argument("--batch_size", type=int, default=None, help="Override eval batch size.")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    p.add_argument("--pin_memory", action="store_true", help="Pin memory for DataLoader.")
    p.add_argument("--skip_latency", action="store_true", help="Skip profiling latency (faster).")
    p.add_argument("--latency_batch_size", type=int, default=1, help="Batch size for latency measurement.")
    args = p.parse_args()

    dataset = str(args.dataset).lower()
    if dataset != "cifar100":
        raise ValueError("This script is intended for CIFAR-100 re-evaluation.")

    output_root = Path(args.output_root)
    dataset_root = Path(args.dataset_root)
    template_cfg = load_config(args.config)

    # Make sure cfg points to the provided dataset root.
    template_cfg["dataset_root"] = str(dataset_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_res = 32
    num_classes = 100
    batch_size = int(args.batch_size) if args.batch_size is not None else int(template_cfg.get("batch_size", 64))
    pin_memory = bool(args.pin_memory)

    criterion = nn.CrossEntropyLoss()

    # Profile once per variant.
    profiles: dict[str, dict[str, Any]] = {}
    for variant in args.variants:
        profiles[variant.lower()] = _profile_variant(
            variant=str(variant).lower(),
            cfg=template_cfg,
            device=device,
            input_res=input_res,
            num_classes=num_classes,
            batch_size_latency=int(args.latency_batch_size),
            skip_latency=bool(args.skip_latency),
        )

    report_rows: list[dict[str, Any]] = []

    for seed in args.seeds:
        for variant in args.variants:
            variant_lower = str(variant).lower()
            run_dir = _run_dir(output_root, dataset=dataset, variant=variant_lower, seed=int(seed))
            ckpt_path = run_dir / "checkpoints" / "best.pt"
            metrics_path = run_dir / "metrics.json"

            if not ckpt_path.exists():
                print(f"[skip] Missing best checkpoint: {ckpt_path}")
                continue
            if not metrics_path.exists():
                print(f"[skip] Missing metrics.json: {metrics_path}")
                continue

            # Build test loader (prefer split metadata -> deterministic transforms).
            cfg = dict(template_cfg)
            cfg["model"] = variant_lower
            cfg["seed"] = int(seed)
            artifacts_dir = run_dir / "artifacts"
            test_loader = _build_test_loader(
                cfg=cfg,
                dataset_root=dataset_root,
                seed=int(seed),
                artifacts_dir=artifacts_dir,
                batch_size=batch_size,
                num_workers=int(args.num_workers),
                pin_memory=pin_memory,
            )

            # Build model and load checkpoint weights.
            model = build_model(cfg).to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model_state_dict"], strict=True)

            test_loss, top1_acc, top5_acc = evaluate_top1_top5(
                model=model,
                loader=test_loader,
                criterion=criterion,
                device=device,
            )

            # Update metrics.json in-place.
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics.setdefault("test", {})
            metrics["test"]["loss"] = float(test_loss)
            metrics["test"]["acc"] = float(top1_acc)
            metrics["test"]["top1_pp"] = 100.0 * float(top1_acc)
            metrics["test"]["top5_pp"] = 100.0 * float(top5_acc)
            metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            row = {
                "dataset": dataset,
                "variant": variant_lower,
                "seed": int(seed),
                "test": {
                    "top1_pp": metrics["test"]["top1_pp"],
                    "top5_pp": metrics["test"]["top5_pp"],
                },
                "profile": profiles[variant_lower],
                "metrics_path": str(metrics_path),
            }
            report_rows.append(row)

            print(
                f"[eval] {dataset}/{variant_lower}/seed_{seed} | "
                f"top1={row['test']['top1_pp']:.2f}% | top5={row['test']['top5_pp']:.2f}%"
            )

    report = {
        "dataset": dataset,
        "input_res": input_res,
        "num_classes": num_classes,
        "seeds": [int(s) for s in args.seeds],
        "variants": [str(v).lower() for v in args.variants],
        "rows": report_rows,
    }

    out_path = output_root / dataset / "re_eval_report.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote report: {out_path}")


if __name__ == "__main__":
    main()

