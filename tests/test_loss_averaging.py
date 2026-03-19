from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from training.evaluate import evaluate
from training.train import train_one_epoch


def _fixed_model() -> torch.nn.Module:
    model = torch.nn.Linear(1, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0], [-1.0]], dtype=torch.float32))
    return model


def _uneven_loader() -> tuple[DataLoader, torch.Tensor, torch.Tensor]:
    features = torch.tensor([[2.0], [2.0], [-2.0]], dtype=torch.float32)
    targets = torch.tensor([0, 0, 0], dtype=torch.long)
    loader = DataLoader(TensorDataset(features, targets), batch_size=2, shuffle=False)
    return loader, features, targets


def _expected_losses(model: torch.nn.Module, features: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    with torch.no_grad():
        logits = model(features)
        per_sample = F.cross_entropy(logits, targets, reduction="none")

    sample_weighted = float(per_sample.mean().item())
    batch_weighted = float((per_sample[:2].mean() + per_sample[2:].mean()).item() / 2.0)
    return sample_weighted, batch_weighted


def test_train_one_epoch_uses_sample_weighted_average_loss() -> None:
    model = _fixed_model()
    loader, features, targets = _uneven_loader()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.0)

    expected_sample_weighted, batch_weighted = _expected_losses(model, features, targets)
    assert expected_sample_weighted != pytest.approx(batch_weighted)

    loss = train_one_epoch(
        model=model,
        train_loader=loader,
        optimizer=optimizer,
        criterion=criterion,
        device=torch.device("cpu"),
        epoch=1,
    )

    assert loss == pytest.approx(expected_sample_weighted)


def test_evaluate_uses_sample_weighted_average_loss() -> None:
    model = _fixed_model()
    loader, features, targets = _uneven_loader()
    criterion = torch.nn.CrossEntropyLoss()

    expected_sample_weighted, batch_weighted = _expected_losses(model, features, targets)
    assert expected_sample_weighted != pytest.approx(batch_weighted)

    loss, _ = evaluate(
        model=model,
        loader=loader,
        criterion=criterion,
        device=torch.device("cpu"),
    )

    assert loss == pytest.approx(expected_sample_weighted)
