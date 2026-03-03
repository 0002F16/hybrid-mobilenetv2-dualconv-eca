"""
Train on CIFAR-10 using config.

Run from project root:
    python experiments/train_cifar10.py
    python experiments/train_cifar10.py --config configs/cifar100.yaml
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.preprocessing import get_dataset_loaders, set_seed
from models.hybrid import HybridMobileNetV2
from training.train import train_one_epoch
from training.evaluate import evaluate
from training.utils import load_config, save_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cifar10.yaml", help="Config path")
    parser.add_argument("--output_dir", default="outputs", help="Output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    kwargs = dict(
        root=cfg["dataset_root"],
        batch_size=cfg["batch_size"],
        num_workers=cfg.get("num_workers", 4),
        seed=cfg["seed"],
    )
    train_loader, val_loader, test_loader = get_dataset_loaders(
        cfg["dataset"], **kwargs
    )

    # Model
    input_size = 32 if "cifar" in cfg["dataset"] else 64
    model = HybridMobileNetV2(
        num_classes=cfg["num_classes"],
        width_multiplier=cfg.get("width_multiplier", 1.0),
        input_size=input_size,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(),
        lr=cfg["learning_rate"],
        momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    summary_log_interval = cfg.get("summary_log_interval_epochs", 10)

    early_cfg = cfg.get("early_stopping", {})
    early_enabled = early_cfg.get("enabled", True)
    warmup_epochs = early_cfg.get("warmup_epochs", 30)
    patience_epochs = early_cfg.get("patience_epochs", 20)
    min_delta_pp = early_cfg.get("min_delta_pp", 0.1)

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    best_acc = -1.0
    best_top1_pp = -1.0
    epochs_without_meaningful_improve = 0

    for epoch in range(1, cfg["epochs"] + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        val_top1_pp = 100.0 * val_acc
        scheduler.step()

        if epoch % summary_log_interval == 0:
            print(
                f"Epoch {epoch}/{cfg['epochs']} | "
                f"Train Loss: {loss:.4f} | Val Loss: {val_loss:.4f} | Val Top-1: {val_top1_pp:.2f}%"
            )

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, loss, ckpt_dir / "best.pt"
            )

        if not early_enabled:
            continue

        if epoch <= warmup_epochs:
            best_top1_pp = max(best_top1_pp, val_top1_pp)
            continue

        meaningful_improvement = (val_top1_pp - best_top1_pp) > min_delta_pp
        if meaningful_improvement:
            best_top1_pp = val_top1_pp
            epochs_without_meaningful_improve = 0
        else:
            epochs_without_meaningful_improve += 1

        if epochs_without_meaningful_improve >= patience_epochs:
            print(
                f"Early stopping at epoch {epoch}: no val Top-1 improvement "
                f"> {min_delta_pp:.3f} pp for {patience_epochs} epochs after warm-up ({warmup_epochs})."
            )
            break

    # Final test
    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Acc: {100*test_acc:.2f}%")


if __name__ == "__main__":
    main()
