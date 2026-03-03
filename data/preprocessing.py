import random
import ssl

# Fix SSL certificate verification on macOS (Python.org installs)
try:
    import certifi

    ssl._create_default_https_context = lambda: ssl.create_default_context(
        cafile=certifi.where()
    )
except ImportError:
    pass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

# ---------------------------------------------------------------------------
# Normalization constants (per-channel mean and std)
# ---------------------------------------------------------------------------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

TINY_IMAGENET_MEAN = (0.485, 0.456, 0.406)
TINY_IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across random, numpy, torch, and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms(dataset_name: str) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Return train and test transforms for the specified dataset.

    Args:
        dataset_name: One of "cifar10", "cifar100", or "tiny_imagenet".

    Returns:
        Tuple of (train_transform, test_transform).

    Raises:
        ValueError: If dataset_name is not recognized.
    """
    name = dataset_name.lower()
    if name == "cifar10":
        mean, std = CIFAR10_MEAN, CIFAR10_STD
        crop_size = 32
    elif name == "cifar100":
        mean, std = CIFAR100_MEAN, CIFAR100_STD
        crop_size = 32
    elif name == "tiny_imagenet":
        mean, std = TINY_IMAGENET_MEAN, TINY_IMAGENET_STD
        crop_size = 64
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            "Choose from 'cifar10', 'cifar100', or 'tiny_imagenet'."
        )

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(crop_size, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    return train_transform, test_transform


class _TransformSubset(Dataset):
    """Wrapper that applies a transform to a subset (e.g., from random_split)."""

    def __init__(
        self,
        subset: torch.utils.data.Subset,
        transform: transforms.Compose,
    ) -> None:
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, label = self.subset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_cifar10_loaders(
    root: str = "./data",
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders for CIFAR-10.

    A 10% validation split is created from the official training set using
    random_split with the given seed.

    Args:
        root: Root directory for dataset download/storage.
        batch_size: Batch size for all loaders.
        num_workers: Number of DataLoader workers.
        pin_memory: Whether to pin memory for faster GPU transfer.
        seed: Random seed for reproducible train/val split.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    set_seed(seed)
    train_transform, test_transform = get_transforms("cifar10")

    full_train = CIFAR10(root=root, train=True, download=True, transform=None)
    test_dataset = CIFAR10(
        root=root, train=False, download=True, transform=test_transform
    )

    n_train = len(full_train)
    n_val = int(0.1 * n_train)
    n_train_split = n_train - n_val

    train_subset, val_subset = random_split(
        full_train,
        [n_train_split, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    train_dataset = _TransformSubset(train_subset, train_transform)
    val_dataset = _TransformSubset(val_subset, test_transform)

    print(
        f"CIFAR-10 - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def get_cifar100_loaders(
    root: str = "./data",
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders for CIFAR-100.

    A 10% validation split is created from the official training set using
    random_split with the given seed.

    Args:
        root: Root directory for dataset download/storage.
        batch_size: Batch size for all loaders.
        num_workers: Number of DataLoader workers.
        pin_memory: Whether to pin memory for faster GPU transfer.
        seed: Random seed for reproducible train/val split.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    set_seed(seed)
    train_transform, test_transform = get_transforms("cifar100")

    full_train = CIFAR100(root=root, train=True, download=True, transform=None)
    test_dataset = CIFAR100(
        root=root, train=False, download=True, transform=test_transform
    )

    n_train = len(full_train)
    n_val = int(0.1 * n_train)
    n_train_split = n_train - n_val

    train_subset, val_subset = random_split(
        full_train,
        [n_train_split, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    train_dataset = _TransformSubset(train_subset, train_transform)
    val_dataset = _TransformSubset(val_subset, test_transform)

    print(
        f"CIFAR-100 - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


class _TinyImageNetTestDataset(Dataset):
    """Dataset for Tiny-ImageNet test set (unlabeled; returns placeholder -1)."""

    def __init__(self, root: Path, transform: transforms.Compose) -> None:
        self.root = Path(root)
        self.transform = transform
        self.images_dir = self.root / "test" / "images"
        self.samples: list[Path] = []
        if self.images_dir.exists():
            self.samples = sorted(self.images_dir.glob("*.JPEG")) + sorted(
                self.images_dir.glob("*.jpg")
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        from PIL import Image

        img_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, -1


def get_tiny_imagenet_loaders(
    root: str,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders for Tiny-ImageNet.

    Expects directory structure:
        root/train/<wnid>/images/*.JPEG  (or train/<wnid>/*.JPEG)
        root/test/images/*.JPEG (optional; test has no public labels)

    A 10% validation split is created from the official training set using
    random_split with the given seed.

    Args:
        root: Path to the Tiny-ImageNet dataset root.
        batch_size: Batch size for all loaders.
        num_workers: Number of DataLoader workers.
        pin_memory: Whether to pin memory for faster GPU transfer.
        seed: Random seed for reproducible train/val split.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    set_seed(seed)
    root_path = Path(root)
    train_transform, test_transform = get_transforms("tiny_imagenet")

    train_dir = root_path / "train"
    full_train = ImageFolder(root=str(train_dir), transform=None)

    n_train = len(full_train)
    n_val = int(0.1 * n_train)
    n_train_split = n_train - n_val

    train_subset, val_subset = random_split(
        full_train,
        [n_train_split, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    train_dataset = _TransformSubset(train_subset, train_transform)
    val_dataset = _TransformSubset(val_subset, test_transform)

    test_dataset = _TinyImageNetTestDataset(root=root_path, transform=test_transform)

    print(
        f"Tiny-ImageNet - Train: {len(train_dataset)}, Val: {len(val_dataset)}, "
        f"Test: {len(test_dataset)}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def get_dataset_loaders(
    dataset_name: str,
    **kwargs,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Dispatcher to get DataLoaders for the specified dataset.

    Args:
        dataset_name: One of "cifar10", "cifar100", or "tiny_imagenet".
        **kwargs: Passed to the dataset-specific loader (root, batch_size, etc.).

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    dispatcher = {
        "cifar10": get_cifar10_loaders,
        "cifar100": get_cifar100_loaders,
        "tiny_imagenet": get_tiny_imagenet_loaders,
    }
    name = dataset_name.lower()
    if name not in dispatcher:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            "Choose from 'cifar10', 'cifar100', or 'tiny_imagenet'."
        )
    return dispatcher[name](**kwargs)
