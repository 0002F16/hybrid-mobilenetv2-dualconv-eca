"""
Unified training + evaluation runner (Colab-friendly).

Runs one config, saves best-val checkpoint, evaluates test from best checkpoint,
and writes machine-readable metrics JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.download_tiny_imagenet import ensure_tiny_imagenet
from data.preprocessing import DEFAULT_SPLIT_SEED, get_dataset_loaders, set_seed
from models.factory import build_model
from training.evaluate import evaluate_top1_top5
from training.trainer import EarlyStoppingConfig, Trainer
from training.utils import build_scheduler, load_config
from utils.profiling import compute_flops, count_parameters, measure_model_size_mb
from utils.versioning import write_env_info_json


def _input_chw_for_dataset(cfg: dict[str, Any], dataset: str) -> tuple[int, int, int]:
    if cfg.get("input_size") is not None:
        h = int(cfg["input_size"])
        return (3, h, h)
    ds = str(dataset).lower()
    if ds in {"cifar10", "cifar100"}:
        return (3, 32, 32)
    if ds == "tiny_imagenet":
        return (3, 64, 64)
    return (3, 32, 32)


def _run_dir(output_root: Path, cfg: dict[str, Any]) -> Path:
    dataset = str(cfg.get("dataset", "unknown")).lower()
    model = str(cfg.get("model", "baseline")).lower()
    seed = int(cfg.get("seed", 0))
    return output_root / dataset / model / f"seed_{seed}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--output_root", default="outputs", help="Root output directory")
    p.add_argument("--model", default=None, help="Optional model override")
    p.add_argument("--dataset_root", default=None, help="Optional dataset_root override")
    p.add_argument(
        "--tiny_imagenet_url",
        default=None,
        help="If set and dataset==tiny_imagenet, download+extract from this URL.",
    )
    p.add_argument("--max_epochs", type=int, default=None, help="Optional epoch cap")
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Use a small synthetic dataset (no downloads). Intended for quick pipeline tests.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoints/last.pt in the run directory, if present.",
    )
    p.add_argument(
        "--resume_path",
        default=None,
        help="Explicit checkpoint path to resume from (overrides --resume default path).",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg["model"] = args.model
    if args.dataset_root:
        cfg["dataset_root"] = args.dataset_root

    dataset = str(cfg["dataset"]).lower()
    if dataset == "tiny_imagenet" and args.tiny_imagenet_url:
        tiny_root = ensure_tiny_imagenet(url=args.tiny_imagenet_url, data_dir=cfg["dataset_root"])
        cfg["dataset_root"] = str(tiny_root)

    if args.max_epochs is not None:
        cfg["epochs"] = int(min(int(cfg["epochs"]), int(args.max_epochs)))

    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_root = Path(args.output_root)
    run_dir = _run_dir(output_root, cfg)
    ckpt_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    artifacts_dir = run_dir / "artifacts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    write_env_info_json(log_dir / "env.json", repo_root=Path(__file__).resolve().parents[1])
    (log_dir / "config.json").write_text(
        json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Data
    if args.synthetic:
        is_cifar = dataset in {"cifar10", "cifar100"}
        input_size = int(cfg.get("input_size", 32 if is_cifar else 64))
        n = int(cfg.get("synthetic_num_samples", 64))
        x = torch.randn(n, 3, input_size, input_size)
        y = torch.randint(0, int(cfg["num_classes"]), (n,))
        ds = TensorDataset(x, y)
        train_loader = DataLoader(ds, batch_size=int(cfg["batch_size"]), shuffle=False)
        val_loader = DataLoader(ds, batch_size=int(cfg["batch_size"]), shuffle=False)
        test_loader = DataLoader(ds, batch_size=int(cfg["batch_size"]), shuffle=False)
    else:
        train_loader, val_loader, test_loader = get_dataset_loaders(
            dataset,
            root=cfg["dataset_root"],
            batch_size=int(cfg["batch_size"]),
            num_workers=int(cfg.get("num_workers", 4)),
            seed=int(cfg["seed"]),
            split_seed=int(cfg.get("split_seed", DEFAULT_SPLIT_SEED)),
            artifacts_root=artifacts_dir,
            randaugment_num_ops=cfg.get("randaugment_num_ops", None),
            randaugment_magnitude=cfg.get("randaugment_magnitude", None),
            random_erasing_p=float(cfg.get("random_erasing_p", 0.0)),
        )

    # Model
    model = build_model(cfg).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg.get("label_smoothing", 0.0)))
    optimizer = SGD(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        momentum=float(cfg["momentum"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    scheduler = build_scheduler(
        optimizer=optimizer,
        cfg=cfg,
        epochs=int(cfg["epochs"]),
    )

    summary_log_interval = int(cfg.get("summary_log_interval_epochs", 10))
    early_cfg = cfg.get("early_stopping", {}) or {}
    early = EarlyStoppingConfig(
        enabled=bool(early_cfg.get("enabled", True)),
        warmup_epochs=int(early_cfg.get("warmup_epochs", 30)),
        patience_epochs=int(early_cfg.get("patience_epochs", 20)),
        min_delta_pp=float(early_cfg.get("min_delta_pp", 0.1)),
    )
    trainer = Trainer(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        log_dir=log_dir,
        ckpt_dir=ckpt_dir,
        epochs=int(cfg["epochs"]),
        val_interval_epochs=int(cfg.get("val_interval_epochs", 1)),
        summary_log_interval_epochs=summary_log_interval,
        early_stopping=early,
        mix_prob=float(cfg.get("mix_prob", 0.0)),
        mixup_alpha=float(cfg.get("mixup_alpha", 1.0)),
        cutmix_alpha=float(cfg.get("cutmix_alpha", 1.0)),
    )
    resume_state = trainer.maybe_resume(resume=bool(args.resume), resume_path=args.resume_path)
    fit_summary = trainer.fit(resume_state=resume_state)

    # Test from best checkpoint
    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_top1, test_top5 = evaluate_top1_top5(
        model, test_loader, criterion, device
    )

    input_chw = _input_chw_for_dataset(cfg, dataset)
    nparams = count_parameters(model)
    flops_result = compute_flops(model, input_chw, device)
    size_mb = measure_model_size_mb(model)

    metrics = {
        "dataset": dataset,
        "model": str(cfg.get("model", "baseline")).lower(),
        "seed": int(cfg["seed"]),
        "best_val": fit_summary["best_val"],
        "stopped_epoch": fit_summary.get("stopped_epoch", None),
        "model_profile": {
            "params": int(nparams),
            "flops": int(flops_result["flops"]),
            "macs": int(flops_result["macs"]),
            "flops_method": flops_result["method_used"],
            "size_mb": float(round(size_mb, 6)),
            "input_size_chw": list(input_chw),
        },
        "test": {
            "loss": float(test_loss),
            "acc": float(test_top1),
            "top1_pp": 100.0 * float(test_top1),
            "top5_pp": 100.0 * float(test_top5),
        },
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"Saved metrics to: {run_dir / 'metrics.json'}")
    print(
        f"Test Top-1: {metrics['test']['top1_pp']:.2f}% | Top-5: {metrics['test']['top5_pp']:.2f}% | "
        f"params={metrics['model_profile']['params']:,} | "
        f"FLOPs={metrics['model_profile']['flops']:,} ({metrics['model_profile']['flops_method']}) | "
        f"size={metrics['model_profile']['size_mb']:.4f} MB"
    )


if __name__ == "__main__":
    main()

