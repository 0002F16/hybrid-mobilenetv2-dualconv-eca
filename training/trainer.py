from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from training.evaluate import evaluate
from training.train import train_one_epoch
from training.utils import load_checkpoint, save_checkpoint


@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    warmup_epochs: int = 30
    patience_epochs: int = 20
    min_delta_pp: float = 0.1


@dataclass
class ResumeState:
    start_epoch: int
    best_val: dict[str, Any]
    best_top1_pp: float
    epochs_without_meaningful_improve: int


class Trainer:
    def __init__(
        self,
        *,
        model: nn.Module,
        device: torch.device,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: LRScheduler | None,
        train_loader: DataLoader,
        val_loader: DataLoader,
        log_dir: Path,
        ckpt_dir: Path,
        epochs: int,
        val_interval_epochs: int = 1,
        summary_log_interval_epochs: int = 10,
        early_stopping: EarlyStoppingConfig | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.log_dir = Path(log_dir)
        self.ckpt_dir = Path(ckpt_dir)
        self.epochs = int(epochs)
        self.val_interval_epochs = int(val_interval_epochs)
        self.summary_log_interval_epochs = int(summary_log_interval_epochs)
        self.early = early_stopping or EarlyStoppingConfig()

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.epochs_jsonl_path = self.log_dir / "epochs.jsonl"

    def _append_epoch_log(self, payload: dict[str, Any]) -> None:
        self.epochs_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.epochs_jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")

    def maybe_resume(self, *, resume: bool, resume_path: str | Path | None = None) -> ResumeState:
        if not resume and resume_path is None:
            return ResumeState(
                start_epoch=1,
                best_val={"val_acc": -1.0, "val_loss": None, "epoch": None},
                best_top1_pp=-1.0,
                epochs_without_meaningful_improve=0,
            )

        path = Path(resume_path) if resume_path is not None else (self.ckpt_dir / "last.pt")
        if not path.exists():
            return ResumeState(
                start_epoch=1,
                best_val={"val_acc": -1.0, "val_loss": None, "epoch": None},
                best_top1_pp=-1.0,
                epochs_without_meaningful_improve=0,
            )

        ckpt = load_checkpoint(path, model=self.model, optimizer=self.optimizer)
        if self.scheduler is not None and "scheduler_state_dict" in ckpt:
            try:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception:
                pass

        state = ckpt.get("trainer_state", {}) or {}
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        return ResumeState(
            start_epoch=max(1, start_epoch),
            best_val=dict(state.get("best_val", {"val_acc": -1.0, "val_loss": None, "epoch": None})),
            best_top1_pp=float(state.get("best_top1_pp", -1.0)),
            epochs_without_meaningful_improve=int(state.get("epochs_without_meaningful_improve", 0)),
        )

    def fit(self, *, resume_state: ResumeState) -> dict[str, Any]:
        best = resume_state.best_val
        best_top1_pp = float(resume_state.best_top1_pp)
        epochs_without_meaningful_improve = int(resume_state.epochs_without_meaningful_improve)

        stopped_epoch: int | None = None

        for epoch in range(int(resume_state.start_epoch), int(self.epochs) + 1):
            train_loss = float(
                train_one_epoch(
                    self.model,
                    self.train_loader,
                    self.optimizer,
                    self.criterion,
                    self.device,
                    epoch,
                )
            )

            val_loss: float | None = None
            val_acc: float | None = None
            val_top1_pp: float | None = None
            is_best = False

            if epoch % self.val_interval_epochs == 0:
                val_loss_f, val_acc_f = evaluate(self.model, self.val_loader, self.criterion, self.device)
                val_loss = float(val_loss_f)
                val_acc = float(val_acc_f)
                val_top1_pp = 100.0 * float(val_acc_f)

                if float(val_acc) > float(best.get("val_acc", -1.0)):
                    best = {"val_acc": float(val_acc), "val_loss": float(val_loss), "epoch": int(epoch)}
                    is_best = True
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        epoch,
                        train_loss,
                        self.ckpt_dir / "best.pt",
                        scheduler=self.scheduler,
                        extra={"trainer_state": {
                            "best_val": best,
                            "best_top1_pp": best_top1_pp,
                            "epochs_without_meaningful_improve": epochs_without_meaningful_improve,
                        }},
                    )

            if self.scheduler is not None:
                self.scheduler.step()

            lr = float(self.optimizer.param_groups[0].get("lr", 0.0))
            self._append_epoch_log(
                {
                    "epoch": int(epoch),
                    "train_loss": float(train_loss),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_top1_pp": val_top1_pp,
                    "lr": lr,
                    "is_best": bool(is_best),
                    "early_stop_counter": int(epochs_without_meaningful_improve),
                }
            )

            save_checkpoint(
                self.model,
                self.optimizer,
                epoch,
                train_loss,
                self.ckpt_dir / "last.pt",
                scheduler=self.scheduler,
                extra={
                    "trainer_state": {
                        "best_val": best,
                        "best_top1_pp": best_top1_pp,
                        "epochs_without_meaningful_improve": epochs_without_meaningful_improve,
                    }
                },
            )

            if epoch % self.summary_log_interval_epochs == 0:
                if val_top1_pp is not None and val_loss is not None:
                    print(
                        f"Epoch {epoch}/{self.epochs} | "
                        f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Top-1: {val_top1_pp:.2f}%"
                    )
                else:
                    print(f"Epoch {epoch}/{self.epochs} | Train Loss: {train_loss:.4f}")

            if not self.early.enabled or val_top1_pp is None:
                continue

            if epoch <= int(self.early.warmup_epochs):
                best_top1_pp = max(best_top1_pp, float(val_top1_pp))
                continue

            meaningful_improvement = (float(val_top1_pp) - float(best_top1_pp)) > float(self.early.min_delta_pp)
            if meaningful_improvement:
                best_top1_pp = float(val_top1_pp)
                epochs_without_meaningful_improve = 0
            else:
                epochs_without_meaningful_improve += 1

            if epochs_without_meaningful_improve >= int(self.early.patience_epochs):
                print(
                    f"Early stopping at epoch {epoch}: no val Top-1 improvement "
                    f"> {float(self.early.min_delta_pp):.3f} pp for {int(self.early.patience_epochs)} "
                    f"epochs after warm-up ({int(self.early.warmup_epochs)})."
                )
                stopped_epoch = int(epoch)
                break

        return {
            "best_val": best,
            "stopped_epoch": stopped_epoch,
        }

