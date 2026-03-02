"""
Example usage of the data preprocessing pipeline.

Run from the project root:
    python example_usage.py
"""

from data.preprocessing import get_cifar10_loaders, get_cifar100_loaders
from data.preprocessing import get_dataset_loaders, get_tiny_imagenet_loaders


def main() -> None:
    # Option 1: Via dispatcher (CIFAR-10)
    print("--- CIFAR-10 via dispatcher ---")
    train_loader, val_loader, test_loader = get_dataset_loaders(
        "cifar10",
        root="./data",
        batch_size=64,
        num_workers=4,
        seed=42,
    )
    batch_x, batch_y = next(iter(train_loader))
    print(f"Batch shape: {batch_x.shape}, labels shape: {batch_y.shape}\n")

    # Option 2: Direct CIFAR-10 (uses default batch_size=64)
    print("--- CIFAR-10 direct ---")
    train_loader, val_loader, test_loader = get_cifar10_loaders(seed=42)
    print()

    # Option 3: CIFAR-100 via dispatcher
    print("--- CIFAR-100 ---")
    train_loader, val_loader, test_loader = get_dataset_loaders(
        "cifar100",
        root="./data",
        batch_size=64,
        seed=42,
    )
    print()

    # Option 4: Tiny-ImageNet (root required; skip if dataset not available)
    tiny_imagenet_path = "/path/to/tiny-imagenet-200"
    try:
        from pathlib import Path
        if Path(tiny_imagenet_path).exists():
            print("--- Tiny-ImageNet ---")
            train_loader, val_loader, test_loader = get_dataset_loaders(
                "tiny_imagenet",
                root=tiny_imagenet_path,
                batch_size=64,
                seed=42,
            )
        else:
            print("--- Tiny-ImageNet ---")
            print(f"Skipped: Set tiny_imagenet_path and download dataset to {tiny_imagenet_path}")
    except Exception as e:
        print(f"Skipped Tiny-ImageNet: {e}")


if __name__ == "__main__":
    main()
