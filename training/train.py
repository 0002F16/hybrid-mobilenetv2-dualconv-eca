"""Training loop."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.mix import maybe_apply_mix


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    *,
    mix_prob: float = 0.0,
    mixup_alpha: float = 1.0,
    cutmix_alpha: float = 1.0,
) -> float:
    """Run one training epoch and return average loss."""
    model.train()
    total_loss = 0.0
    total_examples = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        mixed = maybe_apply_mix(
            data,
            target,
            mix_prob=float(mix_prob),
            mixup_alpha=float(mixup_alpha),
            cutmix_alpha=float(cutmix_alpha),
            cutmix_prob=0.5,
        )
        if mixed is None:
            output = model(data)
            loss = criterion(output, target)
        else:
            x_mixed, y_a, y_b, lam = mixed
            output = model(x_mixed)
            loss = float(lam) * criterion(output, y_a) + (1.0 - float(lam)) * criterion(output, y_b)
        loss.backward()
        optimizer.step()
        batch_size = target.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)
