"""
Tiny-ImageNet dataset loader.

Thin wrapper around data.preprocessing.
"""

from typing import Tuple

from torch.utils.data import DataLoader

from data.preprocessing import get_tiny_imagenet_loaders as _get_tiny_imagenet_loaders


def get_tiny_imagenet_loaders(
    root: str,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
    split_seed: int = 1337,
    artifacts_root: str = "artifacts",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders for Tiny-ImageNet."""
    return _get_tiny_imagenet_loaders(
        root=root,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
        split_seed=split_seed,
        artifacts_root=artifacts_root,
    )
