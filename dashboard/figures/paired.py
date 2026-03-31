"""Tab 5: paired difference strip plots (3 datasets × 3 variants)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.constants import DATASET_ORDER, SEEDS
from dashboard.utils.colors import axis_tick_font, plotly_font_kwargs, plotly_template

NON_BASELINE = ["DualConv-only", "ECA-only", "Hybrid"]


def build_paired_matrix(
    df_runs,
    datasets: list[str],
    theme: str,
):
    """Returns (figure, x_min, x_max)."""
    tpl = plotly_template(theme)  # type: ignore[arg-type]
    ds_list = [d for d in DATASET_ORDER if d in datasets]

    diffs_all = []
    for d in ds_list:
        for v in NON_BASELINE:
            for s in SEEDS:
                b = df_runs[
                    (df_runs["dataset"] == d)
                    & (df_runs["variant"] == "Baseline")
                    & (df_runs["seed"] == s)
                ]["top1_acc"].values
                vv = df_runs[
                    (df_runs["dataset"] == d)
                    & (df_runs["variant"] == v)
                    & (df_runs["seed"] == s)
                ]["top1_acc"].values
                if len(b) and len(vv):
                    diffs_all.append(float(vv[0] - b[0]))
    if diffs_all:
        pad = max(0.5, (max(diffs_all) - min(diffs_all)) * 0.12)
        xr = (min(diffs_all) - pad, max(diffs_all) + pad)
    else:
        xr = (-2.0, 2.0)

    titles = [f"{v} vs Baseline — {d}" for v in NON_BASELINE for d in ds_list]
    rng = np.random.default_rng(42)

    fig = make_subplots(
        rows=3,
        cols=len(ds_list),
        shared_xaxes=True,
        shared_yaxes=False,
        vertical_spacing=0.09,
        horizontal_spacing=0.05,
        subplot_titles=titles,
    )

    for ri, v in enumerate(NON_BASELINE, start=1):
        for ci, d in enumerate(ds_list, start=1):
            xs = []
            seeds = []
            for s in SEEDS:
                b = df_runs[
                    (df_runs["dataset"] == d)
                    & (df_runs["variant"] == "Baseline")
                    & (df_runs["seed"] == s)
                ]["top1_acc"].values
                vv = df_runs[
                    (df_runs["dataset"] == d)
                    & (df_runs["variant"] == v)
                    & (df_runs["seed"] == s)
                ]["top1_acc"].values
                if len(b) and len(vv):
                    xs.append(float(vv[0] - b[0]))
                    seeds.append(s)
            med = float(np.median(xs)) if xs else 0.0
            ys = rng.uniform(-0.15, 0.15, size=len(xs))
            cols = ["#2ca02c" if x > 0 else "#d62728" for x in xs]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers",
                    marker=dict(size=11, color=cols, line=dict(width=1, color="white")),
                    text=[f"seed {s}" for s in seeds],
                    hovertemplate="%{text}<br>Δ Top-1: %{x:.3f} pp<extra></extra>",
                    showlegend=False,
                ),
                row=ri,
                col=ci,
            )
            fig.add_vline(x=0, line_color="black", line_width=1, row=ri, col=ci)
            fig.add_vline(
                x=med,
                line_dash="dot",
                line_color="navy",
                line_width=2,
                row=ri,
                col=ci,
            )
            fig.update_yaxes(range=[-0.35, 0.35], showticklabels=False, row=ri, col=ci)
            fig.update_xaxes(range=list(xr), row=ri, col=ci)

    fig.update_layout(
        template=tpl,
        **plotly_font_kwargs(),
        height=780,
        width=1200,
        title=dict(
            text="Paired Δ Top-1 vs Baseline (percentage points)",
            font=dict(family="Arial", size=12),
        ),
    )
    fig.update_xaxes(title_text="Δ Top-1 (pp)", **axis_tick_font())
    return fig, xr[0], xr[1]
