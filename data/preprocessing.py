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
from typing import Any, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

# ---------------------------------------------------------------------------
# PRD / Phase 2: fixed split seed per dataset
# ---------------------------------------------------------------------------
DEFAULT_SPLIT_SEED = 1337


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


def _ensure_artifacts_split_metadata_dir(artifacts_root: str | Path) -> Path:
    p = Path(artifacts_root) / "split_metadata"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sha256_of_int_list(values: list[int]) -> str:
    import hashlib

    m = hashlib.sha256()
    # stable encoding
    m.update(",".join(map(str, values)).encode("utf-8"))
    return m.hexdigest()


def make_train_val_split_indices(
    n: int, val_fraction: float, split_seed: int
) -> tuple[list[int], list[int]]:
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0,1), got {val_fraction}")
    n_val = int(round(val_fraction * n))
    n_val = max(1, min(n - 1, n_val))
    g = torch.Generator().manual_seed(int(split_seed))
    perm = torch.randperm(n, generator=g).tolist()
    val_indices = perm[:n_val]
    train_indices = perm[n_val:]
    return train_indices, val_indices


def _build_stats_transform(dataset_name: str) -> transforms.Compose:
    name = dataset_name.lower()
    if name in {"cifar10", "cifar100"}:
        # deterministic, no augmentation
        return transforms.Compose([transforms.ToTensor()])
    if name == "tiny_imagenet":
        # deterministic resize/crop to match training resolution target
        return transforms.Compose(
            [
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
            ]
        )
    raise ValueError(f"Unknown dataset for stats transform: {dataset_name}")


@torch.no_grad()
def compute_mean_std_from_dataset(
    dataset: Dataset,
    batch_size: int = 256,
    num_workers: int = 0,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    Compute per-channel mean/std from a dataset that returns (C,H,W) float tensors.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    n_pixels = 0
    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sumsq = torch.zeros(3, dtype=torch.float64)

    for x, _y in loader:
        if not torch.is_tensor(x):
            raise TypeError("Dataset must yield torch.Tensor images for stats")
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected batch shape (N,3,H,W), got {tuple(x.shape)}")
        x = x.to(dtype=torch.float64)
        n = x.shape[0] * x.shape[2] * x.shape[3]
        n_pixels += int(n)
        channel_sum += x.sum(dim=(0, 2, 3))
        channel_sumsq += (x * x).sum(dim=(0, 2, 3))

    mean = channel_sum / n_pixels
    var = (channel_sumsq / n_pixels) - (mean * mean)
    std = torch.sqrt(torch.clamp(var, min=0.0))
    mean_t = (float(mean[0]), float(mean[1]), float(mean[2]))
    std_t = (float(std[0]), float(std[1]), float(std[2]))
    return mean_t, std_t


def get_transforms(
    dataset_name: str,
    *,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    randaugment_num_ops: int | None = None,
    randaugment_magnitude: int | None = None,
    random_erasing_p: float = 0.0,
) -> Tuple[transforms.Compose, transforms.Compose]:
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
    if name not in {"cifar10", "cifar100", "tiny_imagenet"}:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            "Choose from 'cifar10', 'cifar100', or 'tiny_imagenet'."
        )

    if name in {"cifar10", "cifar100"}:
        # PRD 4.4: pad -> random crop (32x32) -> flip -> normalize
        train_steps: list[transforms.Transform] = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
        if randaugment_num_ops is not None and randaugment_magnitude is not None:
            train_steps.append(
                transforms.RandAugment(
                    num_ops=int(randaugment_num_ops),
                    magnitude=int(randaugment_magnitude),
                )
            )
        train_steps.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        if float(random_erasing_p) > 0.0:
            train_steps.append(
                transforms.RandomErasing(
                    p=float(random_erasing_p),
                    scale=(0.02, 0.2),
                    ratio=(0.3, 3.3),
                    value=0,
                )
            )
        train_transform = transforms.Compose(train_steps)
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        # PRD 4.4: resize -> random crop -> flip -> normalize
        train_steps: list[transforms.Transform] = [
            transforms.Resize(64),
            transforms.RandomCrop(64),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
        if randaugment_num_ops is not None and randaugment_magnitude is not None:
            train_steps.append(
                transforms.RandAugment(
                    num_ops=int(randaugment_num_ops),
                    magnitude=int(randaugment_magnitude),
                )
            )
        train_steps.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        if float(random_erasing_p) > 0.0:
            train_steps.append(
                transforms.RandomErasing(
                    p=float(random_erasing_p),
                    scale=(0.02, 0.2),
                    ratio=(0.3, 3.3),
                    value=0,
                )
            )
        train_transform = transforms.Compose(train_steps)
        # Deterministic eval preprocessing: resize + center crop
        test_transform = transforms.Compose(
            [
                transforms.Resize(64),
                transforms.CenterCrop(64),
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


def _validate_label_range(dataset_name: str, labels: list[int]) -> None:
    name = dataset_name.lower()
    if name == "cifar10":
        num_classes = 10
    elif name == "cifar100":
        num_classes = 100
    elif name == "tiny_imagenet":
        num_classes = 200
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    if not labels:
        raise ValueError("No labels provided for validation")
    lo = min(labels)
    hi = max(labels)
    if lo < 0 or hi >= num_classes:
        raise ValueError(
            f"{dataset_name}: label range invalid: min={lo}, max={hi}, expected [0,{num_classes-1}]"
        )


def _write_split_metadata_json(
    *,
    artifacts_root: str | Path,
    dataset: str,
    split_seed: int,
    train_indices: list[int],
    val_indices: list[int],
    train_count: int,
    val_count: int,
    test_count: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    extra: Optional[dict[str, Any]] = None,
) -> Path:
    import json

    out_dir = _ensure_artifacts_split_metadata_dir(artifacts_root)
    out_path = out_dir / f"{dataset.lower()}.json"
    payload: dict[str, Any] = {
        "dataset": dataset.lower(),
        "split_seed": int(split_seed),
        "train_count": int(train_count),
        "val_count": int(val_count),
        "test_count": int(test_count),
        "train_indices_sha256": _sha256_of_int_list(train_indices),
        "val_indices_sha256": _sha256_of_int_list(val_indices),
        "mean": list(mean),
        "std": list(std),
    }
    if extra:
        payload.update(extra)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return out_path


def get_cifar10_loaders(
    root: str = "./data",
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
    split_seed: int = DEFAULT_SPLIT_SEED,
    artifacts_root: str | Path = "artifacts",
    randaugment_num_ops: int | None = None,
    randaugment_magnitude: int | None = None,
    random_erasing_p: float = 0.0,
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
    full_train = CIFAR10(root=root, train=True, download=True, transform=None)
    n = len(full_train)
    train_indices, val_indices = make_train_val_split_indices(
        n=n, val_fraction=0.1, split_seed=split_seed
    )
    train_subset = Subset(full_train, train_indices)
    val_subset = Subset(full_train, val_indices)

    # Compute normalization from post-split training subset only (PRD 4.5)
    stats_transform = _build_stats_transform("cifar10")
    train_stats_ds = _TransformSubset(train_subset, stats_transform)
    mean, std = compute_mean_std_from_dataset(
        train_stats_ds, batch_size=min(256, batch_size), num_workers=0
    )
    train_transform, test_transform = get_transforms(
        "cifar10",
        mean=mean,
        std=std,
        randaugment_num_ops=randaugment_num_ops,
        randaugment_magnitude=randaugment_magnitude,
        random_erasing_p=random_erasing_p,
    )

    test_dataset = CIFAR10(root=root, train=False, download=True, transform=test_transform)

    train_dataset = _TransformSubset(train_subset, train_transform)
    val_dataset = _TransformSubset(val_subset, test_transform)

    # Validate label range deterministically from the subset targets
    train_labels = [full_train.targets[i] for i in train_indices]
    val_labels = [full_train.targets[i] for i in val_indices]
    _validate_label_range("cifar10", train_labels)
    _validate_label_range("cifar10", val_labels)

    _write_split_metadata_json(
        artifacts_root=artifacts_root,
        dataset="cifar10",
        split_seed=split_seed,
        train_indices=train_indices,
        val_indices=val_indices,
        train_count=len(train_dataset),
        val_count=len(val_dataset),
        test_count=len(test_dataset),
        mean=mean,
        std=std,
    )

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
    split_seed: int = DEFAULT_SPLIT_SEED,
    artifacts_root: str | Path = "artifacts",
    randaugment_num_ops: int | None = None,
    randaugment_magnitude: int | None = None,
    random_erasing_p: float = 0.0,
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
    full_train = CIFAR100(root=root, train=True, download=True, transform=None)
    n = len(full_train)
    train_indices, val_indices = make_train_val_split_indices(
        n=n, val_fraction=0.1, split_seed=split_seed
    )
    train_subset = Subset(full_train, train_indices)
    val_subset = Subset(full_train, val_indices)

    stats_transform = _build_stats_transform("cifar100")
    train_stats_ds = _TransformSubset(train_subset, stats_transform)
    mean, std = compute_mean_std_from_dataset(
        train_stats_ds, batch_size=min(256, batch_size), num_workers=0
    )
    train_transform, test_transform = get_transforms(
        "cifar100",
        mean=mean,
        std=std,
        randaugment_num_ops=randaugment_num_ops,
        randaugment_magnitude=randaugment_magnitude,
        random_erasing_p=random_erasing_p,
    )

    test_dataset = CIFAR100(root=root, train=False, download=True, transform=test_transform)

    train_dataset = _TransformSubset(train_subset, train_transform)
    val_dataset = _TransformSubset(val_subset, test_transform)

    train_labels = [full_train.targets[i] for i in train_indices]
    val_labels = [full_train.targets[i] for i in val_indices]
    _validate_label_range("cifar100", train_labels)
    _validate_label_range("cifar100", val_labels)

    _write_split_metadata_json(
        artifacts_root=artifacts_root,
        dataset="cifar100",
        split_seed=split_seed,
        train_indices=train_indices,
        val_indices=val_indices,
        train_count=len(train_dataset),
        val_count=len(val_dataset),
        test_count=len(test_dataset),
        mean=mean,
        std=std,
    )

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


def _parse_tiny_imagenet_val_annotations(path: Path) -> dict[str, str]:
    """
    Parse Tiny-ImageNet val annotations file: each row starts with image filename and wnid.
    Returns mapping: filename -> wnid.
    """
    mapping: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        mapping[parts[0]] = parts[1]
    return mapping


class _TinyImageNetValAsTestDataset(Dataset):
    """
    Tiny-ImageNet official val/ used as test set (PRD 4.2).

    Supports canonical Tiny-ImageNet structure:
      root/val/images/*.JPEG
      root/val/val_annotations.txt  (filename -> wnid)
    """

    def __init__(
        self, root: Path, transform: transforms.Compose, class_to_idx: dict[str, int]
    ) -> None:
        from PIL import Image  # noqa: F401

        self.root = Path(root)
        self.transform = transform
        self.images_dir = self.root / "val" / "images"
        self.ann_path = self.root / "val" / "val_annotations.txt"
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Tiny-ImageNet val images dir not found: {self.images_dir}")
        if not self.ann_path.exists():
            raise FileNotFoundError(f"Tiny-ImageNet val annotations not found: {self.ann_path}")

        self.filename_to_wnid = _parse_tiny_imagenet_val_annotations(self.ann_path)
        self.class_to_idx = class_to_idx
        self.samples: list[tuple[Path, int]] = []
        for img_path in sorted(self.images_dir.glob("*.JPEG")) + sorted(
            self.images_dir.glob("*.jpg")
        ):
            wnid = self.filename_to_wnid.get(img_path.name)
            if wnid is None:
                continue
            if wnid not in self.class_to_idx:
                continue
            self.samples.append((img_path, int(self.class_to_idx[wnid])))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        from PIL import Image

        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_tiny_imagenet_loaders(
    root: str,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
    split_seed: int = DEFAULT_SPLIT_SEED,
    artifacts_root: str | Path = "artifacts",
    randaugment_num_ops: int | None = None,
    randaugment_magnitude: int | None = None,
    random_erasing_p: float = 0.0,
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

    train_dir = root_path / "train"
    full_train = ImageFolder(root=str(train_dir), transform=None)
    n = len(full_train)
    train_indices, val_indices = make_train_val_split_indices(
        n=n, val_fraction=0.1, split_seed=split_seed
    )
    train_subset = Subset(full_train, train_indices)
    val_subset = Subset(full_train, val_indices)

    stats_transform = _build_stats_transform("tiny_imagenet")
    train_stats_ds = _TransformSubset(train_subset, stats_transform)
    mean, std = compute_mean_std_from_dataset(
        train_stats_ds, batch_size=min(256, batch_size), num_workers=0
    )
    train_transform, test_transform = get_transforms(
        "tiny_imagenet",
        mean=mean,
        std=std,
        randaugment_num_ops=randaugment_num_ops,
        randaugment_magnitude=randaugment_magnitude,
        random_erasing_p=random_erasing_p,
    )

    train_dataset = _TransformSubset(train_subset, train_transform)
    val_dataset = _TransformSubset(val_subset, test_transform)

    # PRD: official val/ is used as the test set
    test_dataset = _TinyImageNetValAsTestDataset(
        root=root_path,
        transform=test_transform,
        class_to_idx=full_train.class_to_idx,
    )

    train_labels = [full_train.targets[i] for i in train_indices]
    val_labels = [full_train.targets[i] for i in val_indices]
    _validate_label_range("tiny_imagenet", train_labels)
    _validate_label_range("tiny_imagenet", val_labels)
    # test labels come from val_annotations
    _validate_label_range("tiny_imagenet", [lbl for _p, lbl in test_dataset.samples])

    _write_split_metadata_json(
        artifacts_root=artifacts_root,
        dataset="tiny_imagenet",
        split_seed=split_seed,
        train_indices=train_indices,
        val_indices=val_indices,
        train_count=len(train_dataset),
        val_count=len(val_dataset),
        test_count=len(test_dataset),
        mean=mean,
        std=std,
        extra={
            "tiny_imagenet_test_policy": "official val/ used as test set",
        },
    )

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
