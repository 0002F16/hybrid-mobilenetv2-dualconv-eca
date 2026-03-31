"""Efficiency radar chart (CIFAR-100 Top-1 + static efficiency)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from dashboard.constants import VARIANT_ORDER
from dashboard.utils.colors import get_theme_colors, plotly_font_kwargs, plotly_template


def build_radar_figure(
    df_runs: pd.DataFrame,
    df_eff: pd.DataFrame,
    theme: str,
    normalized: bool = True,
) -> go.Figure:
    """Five axes: Params, FLOPs, Size, Latency, Top-1 (CIFAR-100)."""
    colors = get_theme_colors(theme)  # type: ignore[arg-type]
    tpl = plotly_template(theme)  # type: ignore[arg-type]
    eff = df_eff.set_index("variant")
    top1_c100 = (
        df_runs[df_runs["dataset"] == "CIFAR-100"]
        .groupby("variant")["top1_acc"]
        .mean()
    )
    baseline = eff.loc["Baseline"]
    b_top1 = float(top1_c100.loc["Baseline"])

    categories = ["Params (M)", "FLOPs (M)", "Size (MB)", "Latency (ms)", "Top-1 CIFAR-100 (%)"]

    fig = go.Figure()
    for v in VARIANT_ORDER:
        if v not in eff.index:
            continue
        row = eff.loc[v]
        t1 = float(top1_c100.loc[v]) if v in top1_c100.index else b_top1
        if normalized:
            r = [
                float(row["params_M"]) / float(baseline["params_M"]),
                float(row["flops_M"]) / float(baseline["flops_M"]),
                float(row["size_mb"]) / float(baseline["size_mb"]),
                float(row["latency_ms"]) / float(baseline["latency_ms"]),
                t1 / b_top1,
            ]
        else:
            r = [
                float(row["params_M"]),
                float(row["flops_M"]),
                float(row["size_mb"]),
                float(row["latency_ms"]),
                t1,
            ]
        fig.add_trace(
            go.Scatterpolar(
                r=r + [r[0]],
                theta=categories + [categories[0]],
                fill="toself",
                fillcolor=colors[v],
                opacity=0.25,
                line=dict(color=colors[v]),
                name=v,
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                tickfont=dict(family="Arial", size=10),
            ),
            angularaxis=dict(tickfont=dict(family="Arial", size=10)),
        ),
        template=tpl,
        **plotly_font_kwargs(),
        title=dict(
            text="Efficiency radar"
            + (" (values ÷ Baseline; Top-1: higher is better outward)" if normalized else " (absolute units)"),
            font=dict(family="Arial", size=12),
        ),
        height=520,
        width=700,
        legend=dict(orientation="v", yanchor="middle", y=0.5),
    )
    return fig
