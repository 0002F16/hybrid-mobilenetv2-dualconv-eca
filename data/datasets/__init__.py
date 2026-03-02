"""Dataset-specific loaders wrapping the preprocessing pipeline."""

from data.datasets.cifar import get_cifar10_loaders, get_cifar100_loaders
from data.datasets.tiny_imagenet import get_tiny_imagenet_loaders

__all__ = [
    "get_cifar10_loaders",
    "get_cifar100_loaders",
    "get_tiny_imagenet_loaders",
]
