"""
CIFAR-10 and CIFAR-100 dataset loaders.

Thin wrapper around data.preprocessing.
"""

from typing import Tuple

from torch.utils.data import DataLoader

from data.preprocessing import get_cifar10_loaders as _get_cifar10_loaders
from data.preprocessing import get_cifar100_loaders as _get_cifar100_loaders


def get_cifar10_loaders(
    root: str = "./data",
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
    split_seed: int = 1337,
    artifacts_root: str = "artifacts",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders for CIFAR-10."""
    return _get_cifar10_loaders(
        root=root,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
        split_seed=split_seed,
        artifacts_root=artifacts_root,
    )


def get_cifar100_loaders(
    root: str = "./data",
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
    split_seed: int = 1337,
    artifacts_root: str = "artifacts",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders for CIFAR-100."""
    return _get_cifar100_loaders(
        root=root,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
        split_seed=split_seed,
        artifacts_root=artifacts_root,
    )
