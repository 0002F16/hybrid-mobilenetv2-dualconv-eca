"""Load trained runs and run single-image inference (top-1 / top-5)."""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

from data.preprocessing import get_transforms

_REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    return _REPO_ROOT


@dataclass(frozen=True)
class TrainedRun:
    """One discovered run directory under the trained-models root."""

    path: Path
    dataset: str
    variant: str
    seed: str

    @property
    def label(self) -> str:
        return f"{self.dataset} / {self.variant} / {self.seed}"

    @property
    def config_path(self) -> Path:
        return self.path / "logs" / "config.json"

    @property
    def checkpoint_path(self) -> Path:
        return self.path / "checkpoints" / "best.pt"


def _parse_seed_dir(name: str) -> str | None:
    if not name.startswith("seed_"):
        return None
    return name


def discover_runs(trained_models_root: Path) -> list[TrainedRun]:
    """
    Find run directories that have best checkpoint, config, and split metadata.
    Expected layout: <root>/<dataset>/<variant>/seed_<n>/
    """
    root = Path(trained_models_root)
    if not root.is_dir():
        return []

    runs: list[TrainedRun] = []
    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name.lower()
        for variant_dir in sorted(dataset_dir.iterdir()):
            if not variant_dir.is_dir():
                continue
            variant = variant_dir.name.lower()
            for run_dir in sorted(variant_dir.glob("seed_*")):
                if not run_dir.is_dir():
                    continue
                seed_part = _parse_seed_dir(run_dir.name)
                if seed_part is None:
                    continue
                cfg_path = run_dir / "logs" / "config.json"
                ckpt_path = run_dir / "checkpoints" / "best.pt"
                if not cfg_path.is_file() or not ckpt_path.is_file():
                    continue
                try:
                    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                ds = str(cfg.get("dataset", dataset)).lower()
                split_path = run_dir / "artifacts" / "split_metadata" / f"{ds}.json"
                if not split_path.is_file():
                    continue
                runs.append(
                    TrainedRun(
                        path=run_dir.resolve(),
                        dataset=ds,
                        variant=variant,
                        seed=seed_part,
                    )
                )
    runs.sort(key=lambda r: (r.dataset, r.variant, r.seed))
    return runs


def load_split_mean_std(split_metadata_path: Path) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    payload = json.loads(split_metadata_path.read_text(encoding="utf-8"))
    mean = tuple(float(x) for x in payload["mean"])
    std = tuple(float(x) for x in payload["std"])
    return mean, std


def get_test_transform(
    cfg: dict[str, Any],
    *,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> transforms.Compose:
    _train_t, test_t = get_transforms(
        str(cfg["dataset"]).lower(),
        mean=mean,
        std=std,
        randaugment_num_ops=cfg.get("randaugment_num_ops", None),
        randaugment_magnitude=cfg.get("randaugment_magnitude", None),
        random_erasing_p=float(cfg.get("random_erasing_p", 0.0)),
    )
    return test_t


def get_inference_transform(
    cfg: dict[str, Any],
    *,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> transforms.Compose:
    """
    Test-time transforms plus resize so arbitrary user uploads match model input size.
    CIFAR models expect 32×32; Tiny ImageNet expects 64×64 (resize+centercrop already in test transform).
    """
    test_t = get_test_transform(cfg, mean=mean, std=std)
    ds = str(cfg["dataset"]).lower()
    if ds in {"cifar10", "cifar100"}:
        return transforms.Compose([transforms.Resize((32, 32)), test_t])
    return test_t


def load_config(run: TrainedRun) -> dict[str, Any]:
    return json.loads(run.config_path.read_text(encoding="utf-8"))


def load_model_for_run(
    run: TrainedRun,
    *,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any], tuple[float, float, float], tuple[float, float, float], transforms.Compose]:
    """Build model, load weights, return config and test transform."""
    # Import here so `discover_runs` works without pulling models when testing discovery only
    from models.factory import build_model

    cfg = load_config(run)
    dataset = str(cfg["dataset"]).lower()
    split_path = run.path / "artifacts" / "split_metadata" / f"{dataset}.json"
    mean, std = load_split_mean_std(split_path)
    test_transform = get_inference_transform(cfg, mean=mean, std=std)

    model = build_model(cfg).to(device)
    ckpt = torch.load(run.checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, cfg, mean, std, test_transform


def preprocess_image(
    pil_image: Image.Image,
    test_transform: transforms.Compose,
) -> torch.Tensor:
    """Return batch (1,3,H,W) float tensor on CPU."""
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    x = test_transform(pil_image)
    return x.unsqueeze(0)


def predict_topk(
    model: nn.Module,
    batch: torch.Tensor,
    device: torch.device,
    *,
    k: int = 5,
) -> list[tuple[int, float]]:
    """Return top-k (class_index, probability) with probabilities summing to ~1 over full logits."""
    model.eval()
    with torch.no_grad():
        logits = model(batch.to(device))
        num_classes = int(logits.shape[1])
        k = min(int(k), num_classes)
        probs = torch.softmax(logits, dim=1)
        top_p, top_idx = probs.topk(k=k, dim=1, largest=True, sorted=True)
    row_p = top_p[0].cpu().tolist()
    row_i = top_idx[0].cpu().tolist()
    return list(zip(row_i, row_p))


@functools.lru_cache(maxsize=8)
def _cifar10_class_tuple(*, data_root: str) -> tuple[str, ...]:
    ds = CIFAR10(root=data_root, train=True, download=True)
    return tuple(ds.classes)


@functools.lru_cache(maxsize=8)
def _cifar100_class_tuple(*, data_root: str) -> tuple[str, ...]:
    ds = CIFAR100(root=data_root, train=True, download=True)
    return tuple(ds.classes)


def get_cifar10_classes(*, data_root: Path) -> list[str]:
    return list(_cifar10_class_tuple(data_root=str(Path(data_root).resolve())))


def get_cifar100_classes(*, data_root: Path) -> list[str]:
    return list(_cifar100_class_tuple(data_root=str(Path(data_root).resolve())))


def get_tiny_imagenet_class_names(*, dataset_root: Path) -> list[str] | None:
    """
    Return class folder names in ImageFolder order (sorted wnids under train/), or None if invalid.
    """
    train_dir = Path(dataset_root) / "train"
    if not train_dir.is_dir():
        return None
    return sorted([p.name for p in train_dir.iterdir() if p.is_dir()])


@functools.lru_cache(maxsize=2)
def _tiny_imagenet_label_tuple_from_json(*, labels_json_path: str) -> tuple[str, ...] | None:
    """
    Read `demo/tiny_imagenet_labels.json` which stores a 200-length list of label strings
    aligned with the repo's Tiny-ImageNet training class index convention.
    """
    p = Path(labels_json_path)
    if not p.is_file():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, list) or not payload:
        return None
    if not all(isinstance(x, str) for x in payload):
        return None
    return tuple(payload)


def resolve_label(
    class_idx: int,
    *,
    dataset: str,
    cifar_data_root: Path | str,
    tiny_imagenet_root: Path | None,
) -> str:
    """Human-readable label, or 'class {idx}' if names unavailable."""
    ds = dataset.lower()
    root = Path(cifar_data_root)
    if ds == "cifar10":
        names = get_cifar10_classes(data_root=root)
        if 0 <= class_idx < len(names):
            return names[class_idx]
    elif ds == "cifar100":
        names = get_cifar100_classes(data_root=root)
        if 0 <= class_idx < len(names):
            return names[class_idx]
    elif ds == "tiny_imagenet":
        if tiny_imagenet_root is not None:
            names = get_tiny_imagenet_class_names(dataset_root=tiny_imagenet_root)
            if names is not None and 0 <= class_idx < len(names):
                return names[class_idx]
        # Fallback: use bundled label strings even without the dataset folder.
        labels = _tiny_imagenet_label_tuple_from_json(
            labels_json_path=str(repo_root() / "demo" / "tiny_imagenet_labels.json")
        )
        if labels is not None and 0 <= class_idx < len(labels):
            return labels[class_idx]
    return f"class {class_idx}"
