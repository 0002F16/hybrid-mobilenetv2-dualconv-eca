"""Wilcoxon signed-rank, Holm–Bonferroni, bootstrap CIs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from dashboard.constants import DATASET_ORDER, SEEDS, VARIANT_ORDER

NON_BASELINE = [v for v in VARIANT_ORDER if v != "Baseline"]


def _paired_diffs(
    df: pd.DataFrame,
    dataset: str,
    variant: str,
    metric: str,
) -> np.ndarray | None:
    """Paired differences (variant - Baseline) aligned by seed."""
    col = "top1_acc" if metric == "top1" else "top5_acc"
    base = df[(df["dataset"] == dataset) & (df["variant"] == "Baseline")].set_index("seed")[
        col
    ]
    var = df[(df["dataset"] == dataset) & (df["variant"] == variant)].set_index("seed")[col]
    aligned = []
    for s in SEEDS:
        if s in base.index and s in var.index:
            aligned.append(float(var.loc[s] - base.loc[s]))
    if len(aligned) < 3:
        return None
    return np.asarray(aligned, dtype=float)


def bootstrap_median_ci(
    diffs: np.ndarray,
    n_boot: int = 1000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Median and 95% bootstrap CI for median paired difference."""
    rng = rng or np.random.default_rng(0)
    med = float(np.median(diffs))
    if len(diffs) < 2:
        return med, med, med
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        boots.append(np.median(sample))
    boots = np.sort(np.array(boots))
    lo = float(np.percentile(boots, 2.5))
    hi = float(np.percentile(boots, 97.5))
    return med, lo, hi


def bootstrap_mean_ci_across_seeds(
    values: np.ndarray,
    n_boot: int = 1000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """95% CI for mean (for bar chart overlay)."""
    rng = rng or np.random.default_rng(1)
    if len(values) == 0:
        return float("nan"), float("nan")
    means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))
    means = np.sort(np.array(means))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def build_table_b(df_runs: pd.DataFrame) -> pd.DataFrame:
    """
    Wilcoxon variant vs Baseline × dataset × metric (top1, top5).
    Holm correction; bootstrap CI on median paired difference.
    """
    rows = []
    raw_ps = []

    for metric in ("top1", "top5"):
        for dataset in DATASET_ORDER:
            for variant in NON_BASELINE:
                diffs = _paired_diffs(df_runs, dataset, variant, metric)
                if diffs is None or len(diffs) < 3:
                    rows.append(
                        {
                            "variant": variant,
                            "dataset": dataset,
                            "metric": metric,
                            "median_delta": np.nan,
                            "ci_lower": np.nan,
                            "ci_upper": np.nan,
                            "W_statistic": np.nan,
                            "raw_p": np.nan,
                            "corrected_p": np.nan,
                            "significant": "",
                        }
                    )
                    raw_ps.append(1.0)
                    continue
                med, lo, hi = bootstrap_median_ci(diffs)
                try:
                    wres = stats.wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")
                    w_stat = float(wres.statistic)
                    p = float(wres.pvalue)
                except ValueError:
                    w_stat = float("nan")
                    p = 1.0
                rows.append(
                    {
                        "variant": variant,
                        "dataset": dataset,
                        "metric": metric,
                        "median_delta": med,
                        "ci_lower": lo,
                        "ci_upper": hi,
                        "W_statistic": w_stat,
                        "raw_p": p,
                        "corrected_p": np.nan,
                        "significant": "",
                    }
                )
                raw_ps.append(p)

    out = pd.DataFrame(rows)
    if len(raw_ps):
        _, p_corr, _, _ = multipletests(raw_ps, method="holm")
        out["corrected_p"] = p_corr
        sig = []
        for p in p_corr:
            if p < 0.001:
                sig.append("***")
            elif p < 0.01:
                sig.append("**")
            elif p < 0.05:
                sig.append("*")
            else:
                sig.append("")
        out["significant"] = sig
    return out


def count_seed_wins(
    df: pd.DataFrame, dataset: str, variant: str, metric: str
) -> tuple[int, int]:
    """How many seeds variant > baseline."""
    col = "top1_acc" if metric == "top1" else "top5_acc"
    base = df[(df["dataset"] == dataset) & (df["variant"] == "Baseline")].set_index("seed")[
        col
    ]
    var = df[(df["dataset"] == dataset) & (df["variant"] == variant)].set_index("seed")[col]
    wins = 0
    total = 0
    for s in SEEDS:
        if s in base.index and s in var.index:
            total += 1
            if float(var.loc[s]) > float(base.loc[s]):
                wins += 1
    return wins, total


def get_corrected_p_for_pair(
    table_b: pd.DataFrame, variant: str, dataset: str, metric: str
) -> float:
    sub = table_b[
        (table_b["variant"] == variant)
        & (table_b["dataset"] == dataset)
        & (table_b["metric"] == metric)
    ]
    if len(sub) == 0:
        return float("nan")
    return float(sub["corrected_p"].iloc[0])
