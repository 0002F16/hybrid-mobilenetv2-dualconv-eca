"""Checkpointing, config loading, and logging utilities."""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
)
import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    epochs: int,
) -> LRScheduler | None:
    """Build LR scheduler from config.

    Supported scheduler values:
    - ``cosine``: CosineAnnealingLR(T_max=epochs), optionally preceded by linear LR warmup.
    - ``none`` / ``null`` / missing: disable scheduler

    Top-level ``lr_warmup_epochs`` (default 0): linearly ramps LR from ``start_factor``
    of base LR to full base LR over that many epochs, then cosine-anneals for the
    remaining ``epochs - lr_warmup_epochs`` steps. This is independent of
    ``early_stopping.warmup_epochs``, which only delays when validation patience
    starts; keep ``early_stopping.warmup_epochs >= lr_warmup_epochs`` when using both.
    """
    raw_scheduler = cfg.get("scheduler", None)
    if raw_scheduler is None:
        return None

    scheduler_name = str(raw_scheduler).strip().lower()
    if scheduler_name in {"", "none", "null"}:
        return None
    if scheduler_name == "cosine":
        total_epochs = int(epochs)
        lr_warmup_epochs = int(cfg.get("lr_warmup_epochs", 0))
        if lr_warmup_epochs < 0:
            raise ValueError(f"lr_warmup_epochs must be >= 0, got {lr_warmup_epochs}")
        if lr_warmup_epochs > 0:
            if lr_warmup_epochs >= total_epochs:
                raise ValueError(
                    f"lr_warmup_epochs ({lr_warmup_epochs}) must be < epochs ({total_epochs})"
                )
            warmup = LinearLR(
                optimizer,
                start_factor=1e-6,
                end_factor=1.0,
                total_iters=lr_warmup_epochs,
            )
            cosine = CosineAnnealingLR(
                optimizer,
                T_max=total_epochs - lr_warmup_epochs,
            )
            return SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[lr_warmup_epochs],
            )
        return CosineAnnealingLR(optimizer, T_max=total_epochs)

    valid = "cosine, none"
    raise ValueError(
        f"Unsupported scheduler '{raw_scheduler}'. Supported values: {valid}."
    )


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
