from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data.preprocessing import get_transforms
from training.train import train_one_epoch


def test_transforms_include_randaugment_and_random_erasing_when_enabled() -> None:
    mean = (0.5, 0.5, 0.5)
    std = (0.2, 0.2, 0.2)
    train_t, _test_t = get_transforms(
        "cifar10",
        mean=mean,
        std=std,
        randaugment_num_ops=2,
        randaugment_magnitude=9,
        random_erasing_p=0.25,
    )
    s = repr(train_t)
    assert "RandAugment" in s
    assert "RandomErasing" in s


def test_train_one_epoch_runs_with_mixup_cutmix_enabled() -> None:
    device = torch.device("cpu")
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 10),
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)

    loss = train_one_epoch(
        model,
        loader,
        optimizer,
        criterion,
        device,
        epoch=1,
        mix_prob=1.0,
        mixup_alpha=1.0,
        cutmix_alpha=1.0,
    )
    assert isinstance(loss, float)
    assert loss == loss  # not NaN

