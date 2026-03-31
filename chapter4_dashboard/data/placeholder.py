from __future__ import annotations

import numpy as np
import pandas as pd

from chapter4_dashboard.utils.constants import DATASET_ORDER, SEEDS, VARIANT_ORDER


def _rng(seed: int = 7) -> np.random.Generator:
    return np.random.default_rng(seed)


def _base_top1_means() -> dict[str, float]:
    return {"CIFAR-10": 92.1, "CIFAR-100": 68.4, "Tiny-ImageNet": 58.2}


def _deltas() -> dict[str, dict[str, float]]:
    return {
        "DualConv-only": {"CIFAR-10": 0.3, "CIFAR-100": 1.1, "Tiny-ImageNet": 1.4},
        "ECA-only": {"CIFAR-10": 0.2, "CIFAR-100": 0.9, "Tiny-ImageNet": 1.1},
        "Hybrid": {"CIFAR-10": 0.6, "CIFAR-100": 2.1, "Tiny-ImageNet": 2.8},
        "Baseline": {"CIFAR-10": 0.0, "CIFAR-100": 0.0, "Tiny-ImageNet": 0.0},
    }


def _top5_offsets() -> dict[str, float]:
    return {"CIFAR-10": 4.5, "CIFAR-100": 12.0, "Tiny-ImageNet": 15.5}


def _efficiency_rows() -> pd.DataFrame:
    rows = [
        ("Baseline", 3.40, 300.0, 13.0, 8.2),
        ("DualConv-only", 3.51, 309.0, 13.4, 8.9),
        ("ECA-only", 3.41, 301.0, 13.1, 8.4),
        ("Hybrid", 3.52, 310.0, 13.5, 9.1),
    ]
    return pd.DataFrame(
        rows, columns=["variant", "params_M", "flops_M", "size_mb", "latency_ms"]
    )


def generate_placeholder() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      df_runs: seed, variant, dataset, top1_acc, top5_acc
      df_efficiency: variant, params_M, flops_M, size_mb, latency_ms
      df_curves: variant, dataset, seed, epoch, train_loss, val_loss, val_top1, is_best_epoch
    """
    g = _rng(11)
    base = _base_top1_means()
    deltas = _deltas()
    top5_off = _top5_offsets()

    run_rows: list[dict] = []
    for ds in DATASET_ORDER:
        for v in VARIANT_ORDER:
            mu = base[ds] + deltas[v][ds]
            for s in SEEDS:
                top1 = mu + float(g.normal(0.0, 0.4))
                top5 = top1 + top5_off[ds] + float(g.normal(0.0, 0.3))
                run_rows.append(
                    {
                        "seed": int(s),
                        "variant": v,
                        "dataset": ds,
                        "top1_acc": float(top1),
                        "top5_acc": float(top5),
                    }
                )
    df_runs = pd.DataFrame(run_rows)

    df_eff = _efficiency_rows()

    # Curves: 200 epochs, exponential loss decay, val_top1 converges.
    epochs = np.arange(1, 201, dtype=int)
    curve_rows: list[dict] = []
    for ds in DATASET_ORDER:
        for v in VARIANT_ORDER:
            # baseline asymptote and hybrid uplift schedule
            base_asym = base[ds] - (2.0 if ds == "CIFAR-10" else 6.0)
            final = base[ds] + deltas[v][ds]
            if v == "Hybrid":
                # ensure surpasses baseline around epoch ~80
                final = base[ds] + deltas[v][ds]
            for s in SEEDS:
                # best epoch varies 100-180
                best_epoch = int(g.integers(100, 181))

                # val curve (sigmoid-ish rise)
                k = 0.06 if ds == "CIFAR-10" else 0.045
                midpoint = 70 if v != "Hybrid" else 65
                val_clean = base_asym + (final - base_asym) * (1.0 / (1.0 + np.exp(-k * (epochs - midpoint))))
                val_noise = g.normal(0.0, 0.25, size=len(epochs))
                val_top1 = val_clean + val_noise

                # losses decay exponentially
                train_loss = 2.2 * np.exp(-epochs / 55.0) + 0.15 + g.normal(0.0, 0.03, size=len(epochs))
                val_loss = 2.5 * np.exp(-epochs / 60.0) + 0.25 + g.normal(0.0, 0.04, size=len(epochs))

                # early stopping: mark best_epoch
                is_best = epochs == best_epoch
                for i, ep in enumerate(epochs):
                    curve_rows.append(
                        {
                            "variant": v,
                            "dataset": ds,
                            "seed": int(s),
                            "epoch": int(ep),
                            "train_loss": float(train_loss[i]),
                            "val_loss": float(val_loss[i]),
                            "val_top1": float(val_top1[i]),
                            "is_best_epoch": bool(is_best[i]),
                        }
                    )
    df_curves = pd.DataFrame(curve_rows)
    return df_runs, df_eff, df_curves

