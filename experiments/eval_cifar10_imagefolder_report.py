from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

import sys
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.preprocessing import compute_train_split_mean_std
from models.factory import build_model


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


@torch.no_grad()
def evaluate_with_details(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    correct_top1 = 0
    correct_top5 = 0

    per_class_total = [0 for _ in range(num_classes)]
    per_class_correct = [0 for _ in range(num_classes)]
    confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)

        batch_size = target.size(0)
        total_loss += criterion(output, target).item() * batch_size
        total_examples += batch_size

        maxk = min(5, output.size(1))
        _, pred_topk = output.topk(k=maxk, dim=1, largest=True, sorted=True)
        pred_top1 = pred_topk[:, 0]
        target_expanded = target.view(-1, 1).expand_as(pred_topk)
        correct = pred_topk.eq(target_expanded)

        correct_top1 += correct[:, 0].sum().item()
        correct_top5 += correct.any(dim=1).sum().item()

        for t, p in zip(target.cpu().tolist(), pred_top1.cpu().tolist()):
            per_class_total[t] += 1
            per_class_correct[t] += int(t == p)
            confusion[t][p] += 1

    avg_loss = total_loss / max(total_examples, 1)
    top1 = correct_top1 / max(total_examples, 1)
    top5 = correct_top5 / max(total_examples, 1)

    per_class_accuracy = {}
    for i in range(num_classes):
        denom = per_class_total[i]
        per_class_accuracy[i] = None if denom == 0 else per_class_correct[i] / denom

    macro_acc = sum(v for v in per_class_accuracy.values() if v is not None) / max(
        sum(v is not None for v in per_class_accuracy.values()), 1
    )

    return {
        "loss": avg_loss,
        "top1": top1,
        "top5": top5,
        "top1_pp": 100.0 * top1,
        "top5_pp": 100.0 * top5,
        "macro_acc": macro_acc,
        "macro_acc_pp": 100.0 * macro_acc,
        "per_class_total": per_class_total,
        "per_class_correct": per_class_correct,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion,
        "num_examples": total_examples,
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _device_from_arg(name: str) -> torch.device:
    name = name.lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _find_runs(checkpoints_root: Path) -> list[Path]:
    return sorted(p.parent.parent for p in checkpoints_root.glob("*/seed_*/checkpoints/best.pt"))


def _canonicalize_per_class(per_class_accuracy: dict[int, float | None]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for idx, name in enumerate(CIFAR10_CLASSES):
        v = per_class_accuracy[idx]
        out[name] = None if v is None else round(100.0 * v, 4)
    return out


def _write_markdown_report(summary: dict[str, Any], out_path: Path) -> None:
    rows = summary["rows"]
    best = summary["best_by_external_top1"]
    best_hybrid = summary["best_hybrid_by_external_top1"]
    dataset_counts = summary["dataset"]["class_counts"]
    preprocessing = summary.get("preprocessing", {})

    lines: list[str] = []
    lines.append("# CIFAR-10 External Dataset Evaluation Report")
    lines.append("")
    lines.append(f"Generated: {summary['generated_at']}")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- Path: `{summary['dataset']['path']}`")
    lines.append(f"- Total images: {summary['dataset']['num_images']}")
    lines.append(f"- Classes: {', '.join(CIFAR10_CLASSES)}")
    if preprocessing.get("resize") is None:
        lines.append("- Preprocessing: no resize, original image sizes preserved, then normalize with official CIFAR-10 train-split mean/std")
    else:
        resize_h, resize_w = preprocessing["resize"]
        lines.append(f"- Preprocessing: resize each image to {resize_h}×{resize_w}, then normalize with official CIFAR-10 train-split mean/std")
    lines.append("")
    lines.append("Class counts:")
    for cls in CIFAR10_CLASSES:
        lines.append(f"- {cls}: {dataset_counts.get(cls, 0)}")
    lines.append("")
    lines.append("## Best checkpoint on external dataset")
    lines.append("")
    lines.append(
        f"- Best overall: **{best['model']} / seed {best['seed']}** | "
        f"Top-1: **{best['external_eval']['top1_pp']:.2f}%** | "
        f"Top-5: **{best['external_eval']['top5_pp']:.2f}%** | "
        f"Macro acc: **{best['external_eval']['macro_acc_pp']:.2f}%**"
    )
    lines.append(
        f"- Best hybrid: **seed {best_hybrid['seed']}** | "
        f"Top-1: **{best_hybrid['external_eval']['top1_pp']:.2f}%** | "
        f"Top-5: **{best_hybrid['external_eval']['top5_pp']:.2f}%** | "
        f"Macro acc: **{best_hybrid['external_eval']['macro_acc_pp']:.2f}%**"
    )
    lines.append("")
    lines.append("## Ranking by external Top-1")
    lines.append("")
    for i, row in enumerate(rows, start=1):
        lines.append(
            f"{i}. {row['model']} / seed {row['seed']} — external Top-1 {row['external_eval']['top1_pp']:.2f}% | "
            f"Top-5 {row['external_eval']['top5_pp']:.2f}% | macro {row['external_eval']['macro_acc_pp']:.2f}% | "
            f"official CIFAR-10 test Top-1 {row['official_test'].get('top1_pp', float('nan')):.2f}%"
        )
    lines.append("")
    lines.append("## Per-class accuracy for best overall checkpoint")
    lines.append("")
    for cls, acc in best['external_eval']['per_class_accuracy_pp'].items():
        if acc is None:
            lines.append(f"- {cls}: n/a")
        else:
            lines.append(f"- {cls}: {acc:.2f}%")
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all trained CIFAR-10 checkpoints on an external ImageFolder dataset.")
    parser.add_argument("--dataset_dir", required=True, help="Path to ImageFolder-style CIFAR-10 dataset.")
    parser.add_argument("--checkpoints_root", default="Trained Models/cifar10", help="Root with <variant>/seed_<n>/checkpoints/best.pt")
    parser.add_argument("--official_data_root", default="data", help="Root for official CIFAR-10 data used to recompute train split mean/std.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    parser.add_argument("--resize", type=int, default=32, help="Resize external images to NxN before evaluation. Set to 0 or a negative value to disable resizing.")
    parser.add_argument("--variants", nargs="*", default=None, help="Optional model variants to evaluate, e.g. hybrid eca")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Optional seeds to evaluate, e.g. 42 123")
    parser.add_argument("--output_dir", default="outputs/external_eval/cifar10_curated")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    checkpoints_root = Path(args.checkpoints_root).expanduser().resolve()
    official_data_root = Path(args.official_data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _device_from_arg(args.device)

    mean, std = compute_train_split_mean_std(
        "cifar10",
        data_root=official_data_root,
        split_seed=1337,
        batch_size=256,
    )

    resize_value = int(args.resize)
    resize_hw: tuple[int, int] | None = None if resize_value <= 0 else (resize_value, resize_value)

    transform_steps = []
    if resize_hw is not None:
        transform_steps.append(transforms.Resize(resize_hw))
    transform_steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose(transform_steps)
    dataset = ImageFolder(root=str(dataset_dir), transform=test_transform)

    if dataset.classes != CIFAR10_CLASSES:
        raise ValueError(
            f"Class order mismatch. Expected {CIFAR10_CLASSES}, got {dataset.classes}. "
            "Rename folders to the canonical CIFAR-10 class names/order."
        )

    class_counts = defaultdict(int)
    for _path, label in dataset.samples:
        class_counts[CIFAR10_CLASSES[label]] += 1

    effective_batch_size = int(args.batch_size)
    if resize_hw is None and effective_batch_size != 1:
        print("[info] resize disabled on variable-size images; forcing batch_size=1 to avoid DataLoader stacking errors.")
        effective_batch_size = 1

    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    criterion = nn.CrossEntropyLoss()
    allowed_variants = None if args.variants is None else {str(v).lower() for v in args.variants}
    allowed_seeds = None if args.seeds is None else {int(s) for s in args.seeds}

    rows: list[dict[str, Any]] = []

    for run_dir in _find_runs(checkpoints_root):
        variant_name = run_dir.parent.name.lower()
        seed_value = int(run_dir.name.replace("seed_", ""))
        if allowed_variants is not None and variant_name not in allowed_variants:
            continue
        if allowed_seeds is not None and seed_value not in allowed_seeds:
            continue

        ckpt_path = run_dir / "checkpoints" / "best.pt"
        metrics_path = run_dir / "metrics.json"
        config_path = run_dir / "logs" / "config.json"

        metrics = _load_json(metrics_path) if metrics_path.exists() else {}
        cfg = _load_json(config_path) if config_path.exists() else {}

        cfg = dict(cfg)
        cfg.setdefault("dataset", "cifar10")
        cfg.setdefault("num_classes", 10)
        cfg.setdefault("width_multiplier", 1.0)
        cfg.setdefault("model", run_dir.parent.name)

        model = build_model(cfg).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)

        external_eval = evaluate_with_details(
            model=model,
            loader=loader,
            criterion=criterion,
            device=device,
            num_classes=10,
        )
        external_eval["per_class_accuracy_pp"] = _canonicalize_per_class(external_eval["per_class_accuracy"])
        external_eval["loss"] = round(float(external_eval["loss"]), 8)
        external_eval["top1"] = round(float(external_eval["top1"]), 8)
        external_eval["top5"] = round(float(external_eval["top5"]), 8)
        external_eval["top1_pp"] = round(float(external_eval["top1_pp"]), 4)
        external_eval["top5_pp"] = round(float(external_eval["top5_pp"]), 4)
        external_eval["macro_acc"] = round(float(external_eval["macro_acc"]), 8)
        external_eval["macro_acc_pp"] = round(float(external_eval["macro_acc_pp"]), 4)

        row = {
            "run_dir": str(run_dir),
            "model": run_dir.parent.name,
            "seed": int(run_dir.name.replace("seed_", "")),
            "checkpoint": str(ckpt_path),
            "official_test": metrics.get("test", {}),
            "best_val": metrics.get("best_val", {}),
            "external_eval": external_eval,
        }
        rows.append(row)
        print(
            f"[eval] {row['model']}/seed_{row['seed']} | "
            f"external top1={external_eval['top1_pp']:.2f}% | top5={external_eval['top5_pp']:.2f}% | "
            f"macro={external_eval['macro_acc_pp']:.2f}%"
        )

        del model, ckpt
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not rows:
        raise ValueError("No runs matched the requested checkpoints/filters.")

    rows.sort(key=lambda r: (r["external_eval"]["top1_pp"], r["external_eval"]["macro_acc_pp"]), reverse=True)

    best = rows[0]
    hybrid_rows = [r for r in rows if r["model"] == "hybrid"]
    best_hybrid = (
        max(hybrid_rows, key=lambda r: (r["external_eval"]["top1_pp"], r["external_eval"]["macro_acc_pp"]))
        if hybrid_rows
        else best
    )

    summary = {
        "generated_at": datetime.now().isoformat(),
        "device": str(device),
        "preprocessing": {
            "resize": None if resize_hw is None else [int(resize_hw[0]), int(resize_hw[1])],
            "effective_batch_size": int(effective_batch_size),
            "notes": (
                "External ImageFolder images keep their original sizes before tensor conversion and CIFAR-10 normalization."
                if resize_hw is None
                else f"External ImageFolder images are resized to {resize_hw[0]}x{resize_hw[1]} before tensor conversion and CIFAR-10 normalization."
            ),
        },
        "normalization": {
            "source": "official CIFAR-10 training split mean/std (split_seed=1337)",
            "mean": [round(float(x), 8) for x in mean],
            "std": [round(float(x), 8) for x in std],
        },
        "dataset": {
            "path": str(dataset_dir),
            "num_images": len(dataset),
            "classes": CIFAR10_CLASSES,
            "class_counts": {k: int(v) for k, v in sorted(class_counts.items())},
        },
        "best_by_external_top1": best,
        "best_hybrid_by_external_top1": best_hybrid,
        "rows": rows,
    }

    json_path = output_dir / "summary.json"
    md_path = output_dir / "report.md"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_markdown_report(summary, md_path)

    print(f"Wrote JSON summary: {json_path}")
    print(f"Wrote Markdown report: {md_path}")


if __name__ == "__main__":
    main()
