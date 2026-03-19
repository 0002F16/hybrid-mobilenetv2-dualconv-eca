"""Checkpointing, config loading, and logging utilities."""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str | Path,
    scheduler: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save training checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        try:
            payload["scheduler_state_dict"] = scheduler.state_dict()
        except Exception:
            pass
    if extra:
        payload.update(extra)
    torch.save(
        payload,
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
