from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from chapter4_dashboard.utils.colors import plotly_font_kwargs, plotly_template
from chapter4_dashboard.utils.constants import SEEDS


def build_paired_delta_grid(
    df_runs: pd.DataFrame, datasets: list[str], variants: list[str], theme: str
) -> go.Figure:
    """
    2 columns (datasets) × 3 rows (variants), dot per seed of Δ Top-1 vs Baseline.
    """
    fig = make_subplots(
        rows=len(variants),
        cols=len(datasets),
        shared_xaxes=True,
        shared_yaxes=False,
        horizontal_spacing=0.10,
        vertical_spacing=0.10,
        subplot_titles=[f"{v} — {ds}" for v in variants for ds in datasets],
    )

    all_deltas = []
    for r, v in enumerate(variants, start=1):
        for c, ds in enumerate(datasets, start=1):
            base = df_runs[(df_runs["dataset"] == ds) & (df_runs["variant"] == "Baseline")][
                ["seed", "top1_acc"]
            ].rename(columns={"top1_acc": "base"})
            var = df_runs[(df_runs["dataset"] == ds) & (df_runs["variant"] == v)][
                ["seed", "top1_acc"]
            ].rename(columns={"top1_acc": "var"})
            m = pd.merge(base, var, on="seed", how="inner")
            m = m[m["seed"].isin(SEEDS)].sort_values("seed")
            if m.empty:
                continue
            deltas = (m["var"].astype(float) - m["base"].astype(float)).to_numpy()
            all_deltas.extend(list(deltas))
            colors = ["#2ca02c" if d > 0 else ("#d62728" if d < 0 else "#999999") for d in deltas]
            y = [f"Seed {int(s)}" for s in m["seed"].tolist()]
            fig.add_trace(
                go.Scatter(
                    x=deltas,
                    y=y,
                    mode="markers",
                    marker=dict(color=colors, size=12),
                    hovertemplate="Δ=%{x:.3f} pp<br>%{y}<extra></extra>",
                    showlegend=False,
                ),
                row=r,
                col=c,
            )
            med = float(np.median(deltas))
            fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="#666666", row=r, col=c)
            fig.add_vline(x=med, line_width=2, line_dash="solid", line_color="#1f77b4", row=r, col=c)

    if all_deltas:
        rng = max(0.5, float(np.max(np.abs(all_deltas))) * 1.2)
        fig.update_xaxes(range=[-rng, rng])

    fig.update_layout(
        template=plotly_template(theme),  # type: ignore[arg-type]
        height=820,
        title="Per-seed paired differences (Δ Top-1 vs Baseline)",
        **plotly_font_kwargs(),
    )
    return fig

