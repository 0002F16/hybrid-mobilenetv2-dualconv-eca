"""Tab 3: accuracy–efficiency scatter (three datasets)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.constants import VARIANT_ORDER
from dashboard.utils.colors import axis_tick_font, get_theme_colors, plotly_font_kwargs, plotly_template


def build_accuracy_efficiency_scatter(
    df_runs: pd.DataFrame,
    df_eff: pd.DataFrame,
    datasets: list[str],
    theme: str,
) -> go.Figure:
    colors = get_theme_colors(theme)  # type: ignore[arg-type]
    tpl = plotly_template(theme)  # type: ignore[arg-type]
    stats = (
        df_runs.groupby(["variant", "dataset"], as_index=False)
        .agg(top1_mean=("top1_acc", "mean"), top1_std=("top1_acc", "std"))
    )
    eff = df_eff.set_index("variant")

    n = len(datasets)
    fig = make_subplots(
        rows=1,
        cols=n,
        subplot_titles=datasets,
        horizontal_spacing=0.06,
    )
    baseline_flops = float(eff.loc["Baseline", "flops_M"])
    baseline_top1_by_ds = {}
    for ds in datasets:
        sub = stats[(stats["dataset"] == ds) & (stats["variant"] == "Baseline")]
        baseline_top1_by_ds[ds] = float(sub["top1_mean"].iloc[0]) if len(sub) else 0.0

    for j, ds in enumerate(datasets, start=1):
        x0, x1 = baseline_flops * 0.9, baseline_flops * 1.1
        y0 = baseline_top1_by_ds.get(ds, 0.0)
        fig.add_shape(
            type="rect",
            x0=0,
            x1=baseline_flops,
            y0=y0,
            y1=100,
            fillcolor="rgba(0,128,0,0.12)",
            line_width=0,
            layer="below",
            row=1,
            col=j,
        )
        fig.add_vline(x=x0, line_dash="dash", line_color="gray", row=1, col=j)
        fig.add_vline(x=x1, line_dash="dash", line_color="gray", row=1, col=j)
        fig.add_hline(y=y0, line_dash="dash", line_color="gray", row=1, col=j)
        for v in VARIANT_ORDER:
            if v not in eff.index:
                continue
            st = stats[(stats["dataset"] == ds) & (stats["variant"] == v)]
            if len(st) == 0:
                continue
            xm = float(eff.loc[v, "flops_M"])
            ym = float(st["top1_mean"].iloc[0])
            yerr = float(st["top1_std"].iloc[0]) if not np.isnan(st["top1_std"].iloc[0]) else 0.0
            sm = float(eff.loc[v, "size_mb"])
            fig.add_trace(
                go.Scatter(
                    x=[xm],
                    y=[ym],
                    mode="markers+text",
                    name=v,
                    legendgroup=v,
                    showlegend=(j == 1),
                    marker=dict(
                        size=max(8, min(28, sm * 1.2)),
                        color=colors[v],
                        line=dict(width=1, color="white"),
                    ),
                    text=[v],
                    textposition="top center",
                    textfont=dict(size=9, family="Arial"),
                    error_y=dict(type="data", array=[yerr], visible=True),
                ),
                row=1,
                col=j,
            )

    fig.update_xaxes(title_text="FLOPs (M)", **axis_tick_font())
    fig.update_yaxes(title_text="Top-1 mean (%)", **axis_tick_font())
    fig.update_layout(
        template=tpl,
        **plotly_font_kwargs(),
        height=520,
        width=1200,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig
