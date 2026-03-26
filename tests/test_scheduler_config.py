from __future__ import annotations

import pytest
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR

from training.utils import build_scheduler


def _optimizer() -> torch.optim.Optimizer:
    model = torch.nn.Linear(4, 2)
    return SGD(model.parameters(), lr=0.1, momentum=0.9)


def test_scheduler_cosine_builds_cosine_annealing() -> None:
    scheduler = build_scheduler(
        optimizer=_optimizer(),
        cfg={"scheduler": "cosine"},
        epochs=10,
    )
    assert isinstance(scheduler, CosineAnnealingLR)


@pytest.mark.parametrize("cfg", [{}, {"scheduler": None}, {"scheduler": "none"}, {"scheduler": "null"}])
def test_scheduler_none_or_missing_returns_none(cfg: dict[str, object]) -> None:
    scheduler = build_scheduler(
        optimizer=_optimizer(),
        cfg=cfg,
        epochs=10,
    )
    assert scheduler is None


def test_scheduler_invalid_name_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported scheduler"):
        build_scheduler(
            optimizer=_optimizer(),
            cfg={"scheduler": "step"},
            epochs=10,
        )


def test_scheduler_cosine_with_lr_warmup_builds_sequential() -> None:
    scheduler = build_scheduler(
        optimizer=_optimizer(),
        cfg={"scheduler": "cosine", "lr_warmup_epochs": 3},
        epochs=10,
    )
    assert isinstance(scheduler, SequentialLR)


def test_scheduler_lr_warmup_ge_epochs_raises() -> None:
    with pytest.raises(ValueError, match="lr_warmup_epochs"):
        build_scheduler(
            optimizer=_optimizer(),
            cfg={"scheduler": "cosine", "lr_warmup_epochs": 10},
            epochs=10,
        )


def test_scheduler_negative_lr_warmup_raises() -> None:
    with pytest.raises(ValueError, match="lr_warmup_epochs"):
        build_scheduler(
            optimizer=_optimizer(),
            cfg={"scheduler": "cosine", "lr_warmup_epochs": -1},
            epochs=10,
        )
