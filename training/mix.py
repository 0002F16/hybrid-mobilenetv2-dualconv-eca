from __future__ import annotations

import math
import random
from typing import Tuple

import torch


def _rand_bbox(*, h: int, w: int, lam: float) -> tuple[int, int, int, int]:
    """Sample a CutMix rectangle. Returns (y1, y2, x1, x2)."""
    cut_ratio = math.sqrt(max(0.0, 1.0 - float(lam)))
    cut_h = int(round(h * cut_ratio))
    cut_w = int(round(w * cut_ratio))

    cy = random.randint(0, h - 1)
    cx = random.randint(0, w - 1)

    y1 = max(0, cy - cut_h // 2)
    y2 = min(h, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(w, cx + cut_w // 2)
    return y1, y2, x1, x2


def mixup(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply Mixup and return (x_mixed, y_a, y_b, lam)."""
    if float(alpha) <= 0.0:
        raise ValueError(f"mixup alpha must be > 0, got {alpha}")
    if x.ndim != 4:
        raise ValueError(f"Expected x shape (N,C,H,W), got {tuple(x.shape)}")
    n = x.size(0)
    perm = torch.randperm(n, device=x.device)
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    x_mixed = lam * x + (1.0 - lam) * x[perm]
    return x_mixed, y, y[perm], lam


def cutmix(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply CutMix and return (x_mixed, y_a, y_b, lam_adjusted)."""
    if float(alpha) <= 0.0:
        raise ValueError(f"cutmix alpha must be > 0, got {alpha}")
    if x.ndim != 4:
        raise ValueError(f"Expected x shape (N,C,H,W), got {tuple(x.shape)}")
    n, _c, h, w = x.shape
    perm = torch.randperm(n, device=x.device)
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())

    y1, y2, x1, x2 = _rand_bbox(h=int(h), w=int(w), lam=lam)
    x_mixed = x.clone()
    x_mixed[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]

    area = float((y2 - y1) * (x2 - x1))
    lam_adjusted = 1.0 - area / float(h * w)
    return x_mixed, y, y[perm], float(lam_adjusted)


def maybe_apply_mix(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    mix_prob: float,
    mixup_alpha: float,
    cutmix_alpha: float,
    cutmix_prob: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float] | None:
    """Optionally apply Mixup/CutMix. Returns None if not applied."""
    if float(mix_prob) <= 0.0:
        return None
    if random.random() >= float(mix_prob):
        return None

    # 50/50 by default
    if random.random() < float(cutmix_prob):
        return cutmix(x, y, alpha=float(cutmix_alpha))
    return mixup(x, y, alpha=float(mixup_alpha))

