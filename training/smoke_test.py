"""Quick smoke test utilities for model/training pipeline."""

from __future__ import annotations

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

from data.preprocessing import set_seed
from models.factory import build_model
from training.trainer import EarlyStoppingConfig, Trainer
from training.utils import build_scheduler, load_config


def run_training_smoke_test(
    config_path: str | Path = "configs/cifar10.yaml",
    batch_size: int = 8,
    device: str | None = None,
) -> dict[str, Any]:
    """
    Run a single synthetic train step to validate the pipeline.

    This test avoids dataset download and does not run a real training loop.
    It verifies that model creation, forward pass, loss computation, backward
    pass, and optimizer step all execute successfully.
    """
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    if device is None:
        resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved_device = torch.device(device)

    input_size = 32 if "cifar" in cfg["dataset"].lower() else 64
    cfg = dict(cfg)
    cfg["input_size"] = int(cfg.get("input_size", input_size))
    model = build_model(cfg).to(resolved_device)

    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg.get("label_smoothing", 0.0)))
    optimizer = SGD(
        model.parameters(),
        lr=cfg["learning_rate"],
        momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"],
    )

    model.train()
    inputs = torch.randn(batch_size, 3, input_size, input_size, device=resolved_device)
    targets = torch.randint(
        0, cfg["num_classes"], (batch_size,), device=resolved_device
    )

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        preds = outputs.argmax(dim=1)
        accuracy = (preds == targets).float().mean().item()

    return {
        "ok": True,
        "dataset": cfg["dataset"],
        "num_classes": cfg["num_classes"],
        "input_size": input_size,
        "batch_size": batch_size,
        "device": str(resolved_device),
        "loss": float(loss.item()),
        "batch_accuracy": float(accuracy),
        "output_shape": tuple(outputs.shape),
    }


def run_end_to_end_mini_run(
    *,
    config_path: str | Path = "configs/cifar10.yaml",
    output_dir: str | Path = "outputs/smoke_mini_run",
    epochs: int = 2,
    batches_per_epoch: int = 2,
    batch_size: int = 8,
    device: str | None = None,
) -> dict[str, Any]:
    """
    End-to-end mini-run using the canonical factory + Trainer on synthetic data.

    This avoids dataset downloads but exercises:
    - models.factory.build_model(cfg)
    - checkpointing (best/last)
    - per-epoch JSONL logs
    - final metrics.json schema
    """
    cfg = load_config(config_path)
    cfg = dict(cfg)
    set_seed(int(cfg["seed"]))

    if device is None:
        resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved_device = torch.device(device)

    dataset_name = str(cfg.get("dataset", "cifar10")).lower()
    input_size = int(cfg.get("input_size", 32 if "cifar" in dataset_name else 64))
    cfg["input_size"] = input_size

    model = build_model(cfg).to(resolved_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        momentum=float(cfg["momentum"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    mini_epochs = int(epochs)
    cfg_sched = dict(cfg)
    w = int(cfg_sched.get("lr_warmup_epochs", 0))
    if w >= mini_epochs:
        cfg_sched["lr_warmup_epochs"] = 0
    scheduler = build_scheduler(
        optimizer=optimizer,
        cfg=cfg_sched,
        epochs=mini_epochs,
    )

    n = int(batch_size) * int(batches_per_epoch)
    x = torch.randn(n, 3, input_size, input_size)
    y = torch.randint(0, int(cfg["num_classes"]), (n,))
    ds = TensorDataset(x, y)
    train_loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False)
    val_loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False)

    out_root = Path(output_dir)
    run_dir = out_root / str(cfg.get("dataset", "unknown")).lower() / str(cfg.get("model", "baseline")).lower() / f"seed_{int(cfg.get('seed', 0))}"
    ckpt_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    (log_dir / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    early_cfg = cfg.get("early_stopping", {}) or {}
    early = EarlyStoppingConfig(
        enabled=bool(early_cfg.get("enabled", True)),
        warmup_epochs=int(early_cfg.get("warmup_epochs", 0)),
        patience_epochs=int(early_cfg.get("patience_epochs", 1)),
        min_delta_pp=float(early_cfg.get("min_delta_pp", 0.0)),
    )

    trainer = Trainer(
        model=model,
        device=resolved_device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        log_dir=log_dir,
        ckpt_dir=ckpt_dir,
        epochs=int(epochs),
        val_interval_epochs=1,
        summary_log_interval_epochs=1,
        early_stopping=early,
        mix_prob=float(cfg.get("mix_prob", 0.0)),
        mixup_alpha=float(cfg.get("mixup_alpha", 1.0)),
        cutmix_alpha=float(cfg.get("cutmix_alpha", 1.0)),
    )
    fit_summary = trainer.fit(resume_state=trainer.maybe_resume(resume=False))

    ckpt = torch.load(ckpt_dir / "best.pt", map_location=resolved_device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    from training.evaluate import evaluate as _evaluate

    test_loss, test_acc = _evaluate(model, val_loader, criterion, resolved_device)

    metrics = {
        "dataset": str(cfg.get("dataset", "unknown")).lower(),
        "model": str(cfg.get("model", "baseline")).lower(),
        "seed": int(cfg.get("seed", 0)),
        "best_val": fit_summary["best_val"],
        "stopped_epoch": fit_summary.get("stopped_epoch", None),
        "test": {"loss": float(test_loss), "acc": float(test_acc), "top1_pp": 100.0 * float(test_acc)},
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "ok": True,
        "run_dir": str(run_dir),
        "epochs_jsonl": str(log_dir / "epochs.jsonl"),
        "metrics_json": str(run_dir / "metrics.json"),
        "best_ckpt": str(ckpt_dir / "best.pt"),
        "last_ckpt": str(ckpt_dir / "last.pt"),
    }


def _format_result(result: dict[str, Any]) -> str:
    return (
        "Smoke test passed | "
        f"dataset={result['dataset']} | "
        f"device={result['device']} | "
        f"input_size={result['input_size']} | "
        f"batch_size={result['batch_size']} | "
        f"loss={result['loss']:.4f} | "
        f"batch_acc={100.0 * result['batch_accuracy']:.2f}% | "
        f"output_shape={result['output_shape']}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run synthetic training smoke test")
    parser.add_argument("--config", default="configs/cifar10.yaml", help="Config path")
    parser.add_argument(
        "--mode",
        choices=["step", "mini_run"],
        default="step",
        help="step: one synthetic optimization step; mini_run: 1-2 epoch end-to-end run via Trainer.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Synthetic batch size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="mini_run only: number of epochs to run",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/smoke_mini_run",
        help="mini_run only: root output directory",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override (e.g., 'cpu' or 'cuda'). Default: auto",
    )
    args = parser.parse_args()

    if args.mode == "step":
        result_dict = run_training_smoke_test(
            config_path=args.config,
            batch_size=args.batch_size,
            device=args.device,
        )
        print(_format_result(result_dict))
    else:
        result_dict = run_end_to_end_mini_run(
            config_path=args.config,
            output_dir=args.output_dir,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            device=args.device,
        )
        print(
            "Mini-run completed | "
            f"run_dir={result_dict['run_dir']} | "
            f"epochs_jsonl={result_dict['epochs_jsonl']} | "
            f"metrics_json={result_dict['metrics_json']}"
        )
