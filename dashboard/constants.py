"""Shared experiment constants (variants, datasets, seeds)."""

SEEDS = [42, 123, 3407, 2024, 777]

VARIANT_ORDER = ["Baseline", "DualConv-only", "ECA-only", "Hybrid"]
VARIANT_FOLDERS = ["baseline", "dualconv", "eca", "hybrid"]

# folder name -> display name
VARIANT_FOLDER_TO_DISPLAY = {
    "baseline": "Baseline",
    "dualconv": "DualConv-only",
    "eca": "ECA-only",
    "hybrid": "Hybrid",
}

DISPLAY_TO_FOLDER = {v: k for k, v in VARIANT_FOLDER_TO_DISPLAY.items()}

DATASET_ORDER = ["CIFAR-10", "CIFAR-100", "Tiny-ImageNet"]

DATASET_FOLDER_TO_DISPLAY = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "tiny_imagenet": "Tiny-ImageNet",
}

DATASET_DISPLAY_TO_FOLDER = {v: k for k, v in DATASET_FOLDER_TO_DISPLAY.items()}

# Trained model tree roots (relative to repo root)
TRAINED_MODEL_ROOTS = ["Trained Models v3", "Trained Models v2", "Trained Models v1"]

# Sidebar / loader: single version at a time (folder name under repo root)
VERSION_TO_FOLDER = {
    "v1": "Trained Models v1",
    "v2": "Trained Models v2",
    "v3": "Trained Models v3",
}

# Static per-variant latency (ms); params/FLOPs/size come from metrics.json on disk
EFFICIENCY_ROWS = [
    ("Baseline", 3.40, 300.0, 13.0, 8.2),
    ("DualConv-only", 3.51, 309.0, 13.4, 8.9),
    ("ECA-only", 3.41, 301.0, 13.1, 8.4),
    ("Hybrid", 3.52, 310.0, 13.5, 9.1),
]


def static_latency_ms(variant_display: str) -> float:
    for v, _p, _f, _s, lat in EFFICIENCY_ROWS:
        if v == variant_display:
            return float(lat)
    return 8.2
