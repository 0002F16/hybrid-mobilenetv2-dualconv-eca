"""Quick smoke test utilities for model/training pipeline."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn as nn
from torch.optim import SGD

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.preprocessing import set_seed
from models.hybrid import HybridMobileNetV2
from training.utils import load_config


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
    model = HybridMobileNetV2(
        num_classes=cfg["num_classes"],
        width_multiplier=cfg.get("width_multiplier", 1.0),
        input_size=input_size,
    ).to(resolved_device)

    criterion = nn.CrossEntropyLoss()
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
        "--batch_size", type=int, default=8, help="Synthetic batch size"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override (e.g., 'cpu' or 'cuda'). Default: auto",
    )
    args = parser.parse_args()

    result_dict = run_training_smoke_test(
        config_path=args.config,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(_format_result(result_dict))
