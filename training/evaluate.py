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


def evaluate_top1_top5(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    Compute loss, Top-1 accuracy, and Top-5 accuracy on the given loader.

    Returns:
        (avg_loss, top1_acc, top5_acc) where accuracies are in [0, 1].
    """
    model.eval()
    total_loss = 0.0
    total_examples = 0

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    maxk = 5
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            batch_size = target.size(0)
            total_loss += criterion(output, target).item() * batch_size
            total_examples += batch_size

            # pred shape: (N, 5) with top-5 class indices per sample
            _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
            target_expanded = target.view(-1, 1).expand_as(pred)
            correct = pred.eq(target_expanded)

            correct_top1 += correct[:, 0].sum().item()
            correct_top5 += correct.any(dim=1).sum().item()
            total += target.size(0)

    avg_loss = total_loss / max(total_examples, 1)
    top1_acc = correct_top1 / max(total, 1)
    top5_acc = correct_top5 / max(total, 1)
    return avg_loss, top1_acc, top5_acc
