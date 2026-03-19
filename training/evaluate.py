"""Validation and test evaluation."""

from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Compute loss and accuracy on the given loader."""
    model.eval()
    total_loss = 0.0
    total_examples = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            batch_size = target.size(0)
            total_loss += criterion(output, target).item() * batch_size
            total_examples += batch_size
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    avg_loss = total_loss / max(total_examples, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy
