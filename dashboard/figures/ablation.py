"""Tab 6: ablation horizontal bars (CIFAR-100, Tiny-ImageNet)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dashboard.utils.colors import axis_tick_font, plotly_font_kwargs, plotly_template


def _mean_delta_vs_baseline(df: pd.DataFrame, dataset: str, variant: str) -> tuple[float, float]:
    col = "top1_acc"
    paired = []
    for s in df["seed"].unique():
        b = df[
            (df["dataset"] == dataset) & (df["variant"] == "Baseline") & (df["seed"] == s)
        ][col].values
        v = df[
            (df["dataset"] == dataset) & (df["variant"] == variant) & (df["seed"] == s)
        ][col].values
        if len(b) and len(v):
            paired.append(float(v[0] - b[0]))
    arr = np.asarray(paired)
    if len(arr) == 0:
        return 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0


def build_ablation_figure(
    df_runs: pd.DataFrame,
    dataset: str,
    color_map: dict[str, str],
    theme: str,
) -> go.Figure:
    tpl = plotly_template(theme)  # type: ignore[arg-type]
    order = ["Baseline", "DualConv-only", "ECA-only", "Hybrid"]
    y_labels = []
    means = []
    errs = []
    for v in order:
        if v == "Baseline":
            y_labels.append(v)
            means.append(0.0)
            errs.append(0.0)
        else:
            m, s = _mean_delta_vs_baseline(df_runs, dataset, v)
            y_labels.append(v)
            means.append(m)
            errs.append(s)

    colors = []
    for m in means:
        if m <= 0:
            colors.append("rgba(150,150,150,0.6)")
        else:
            t = min(1.0, m / 3.0)
            colors.append(f"rgba(0,{120 + int(80 * t)},60,0.85)")

    fig = go.Figure(
        go.Bar(
            orientation="h",
            y=y_labels,
            x=means,
            marker_color=colors,
            error_x=dict(type="data", array=errs, visible=True, color="rgba(0,0,0,0.4)"),
        )
    )
    fig.add_vline(x=0, line_color="black", line_width=1)
    fig.update_layout(
        template=tpl,
        **plotly_font_kwargs(),
        title=dict(text=f"Δ Top-1 vs Baseline — {dataset}", font=dict(family="Arial", size=12)),
        xaxis=dict(title="Mean Δ Top-1 (pp across seeds)", **axis_tick_font()),
        yaxis=dict(**axis_tick_font()),
        height=320,
        width=900,
    )
    return fig


def interaction_metrics(df: pd.DataFrame, dataset: str) -> tuple[float, float, float, str]:
    """DualConv pp, ECA pp, interaction Z, label."""
    d_dual, _ = _mean_delta_vs_baseline(df, dataset, "DualConv-only")
    d_eca, _ = _mean_delta_vs_baseline(df, dataset, "ECA-only")
    d_hyb, _ = _mean_delta_vs_baseline(df, dataset, "Hybrid")
    z = d_hyb - d_dual - d_eca
    if z > 0:
        label = f"Synergistic (+{z:.2f} pp)"
    else:
        label = f"Redundant ({z:.2f} pp)"
    return d_dual, d_eca, z, label
