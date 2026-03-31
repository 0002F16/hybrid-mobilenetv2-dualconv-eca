from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from chapter4_dashboard.utils.constants import DATASET_ORDER, SEEDS


def _bootstrap_ci_median(x: np.ndarray, n_resamples: int = 1000, alpha: float = 0.05, seed: int = 123) -> tuple[float, float]:
    g = np.random.default_rng(seed)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    meds = []
    for _ in range(n_resamples):
        sample = g.choice(x, size=len(x), replace=True)
        meds.append(float(np.median(sample)))
    lo = float(np.quantile(meds, alpha / 2))
    hi = float(np.quantile(meds, 1 - alpha / 2))
    return lo, hi


def _paired_diffs(
    df_runs: pd.DataFrame, dataset: str, variant: str, metric_col: str
) -> np.ndarray:
    base = df_runs[(df_runs["dataset"] == dataset) & (df_runs["variant"] == "Baseline")][
        ["seed", metric_col]
    ].rename(columns={metric_col: "baseline"})
    var = df_runs[(df_runs["dataset"] == dataset) & (df_runs["variant"] == variant)][
        ["seed", metric_col]
    ].rename(columns={metric_col: "variant"})
    merged = pd.merge(base, var, on="seed", how="inner")
    merged = merged[merged["seed"].isin(SEEDS)]
    return (merged["variant"].astype(float) - merged["baseline"].astype(float)).to_numpy()


def compute_table_b(df_runs: pd.DataFrame) -> pd.DataFrame:
    """
    Table B spec:
      - variant ∈ {DualConv-only, ECA-only, Hybrid}
      - dataset ∈ {CIFAR-10, CIFAR-100, Tiny-ImageNet}
      - metric: Top-1 always, Top-5 only for CIFAR-100 and Tiny-ImageNet
      - paired diffs vs baseline across seeds
      - bootstrap CI on median, Wilcoxon signed-rank, Holm correction per metric across 9 rows
    """
    if df_runs.empty:
        return pd.DataFrame(
            columns=[
                "Variant",
                "Dataset",
                "Metric",
                "Median Δ (pp)",
                "95% CI",
                "W stat",
                "Raw p",
                "Corrected p",
                "Sig.",
            ]
        )

    comparisons: list[dict] = []
    variants = ["DualConv-only", "ECA-only", "Hybrid"]
    metric_specs = [("Top-1", "top1_acc"), ("Top-5", "top5_acc")]
    for ds in DATASET_ORDER:
        for v in variants:
            for metric_name, col in metric_specs:
                if metric_name == "Top-5" and ds == "CIFAR-10":
                    continue
                diffs = _paired_diffs(df_runs, ds, v, col)
                med = float(np.median(diffs)) if len(diffs) else float("nan")
                lo, hi = _bootstrap_ci_median(diffs) if len(diffs) else (float("nan"), float("nan"))
                try:
                    w = wilcoxon(diffs, alternative="two-sided").statistic if len(diffs) else float("nan")
                    p = wilcoxon(diffs, alternative="two-sided").pvalue if len(diffs) else float("nan")
                except ValueError:
                    w, p = float("nan"), float("nan")
                comparisons.append(
                    {
                        "Variant": v,
                        "Dataset": ds,
                        "Metric": metric_name,
                        "Median Δ (pp)": med,
                        "ci_lower": lo,
                        "ci_upper": hi,
                        "95% CI": f"[{lo:+.3f}, {hi:+.3f}]",
                        "W stat": w,
                        "Raw p": p,
                    }
                )

    df = pd.DataFrame(comparisons)
    out_parts = []
    for metric_name in df["Metric"].unique():
        sub = df[df["Metric"] == metric_name].copy()
        pvals = sub["Raw p"].to_numpy(dtype=float)
        ok = np.isfinite(pvals)
        corrected = np.full_like(pvals, np.nan, dtype=float)
        if ok.any():
            corrected[ok] = multipletests(pvals[ok], method="holm")[1]
        sub["Corrected p"] = corrected
        sub["Sig."] = sub["Corrected p"].apply(_sig_code)
        out_parts.append(sub)

    out = pd.concat(out_parts, ignore_index=True)
    out["Corrected p"] = out["Corrected p"].astype(float)
    return out[
        [
            "Variant",
            "Dataset",
            "Metric",
            "Median Δ (pp)",
            "95% CI",
            "W stat",
            "Raw p",
            "Corrected p",
            "Sig.",
        ]
    ]


def _sig_code(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""

