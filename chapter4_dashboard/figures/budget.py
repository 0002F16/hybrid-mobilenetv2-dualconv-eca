from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from chapter4_dashboard.utils.colors import plotly_font_kwargs, plotly_template


def build_budget_bars(df_eff: pd.DataFrame, cmap: dict[str, str], theme: str) -> go.Figure:
    """
    Horizontal grouped bars: Params and FLOPs, with dashed ±10% reference vs baseline.
    """
    if df_eff.empty:
        return go.Figure()
    eff = df_eff.set_index("variant")
    if "Baseline" not in eff.index:
        return go.Figure()

    base_p = float(eff.loc["Baseline", "params_M"])
    base_f = float(eff.loc["Baseline", "flops_M"])
    variants = [v for v in ["Baseline", "DualConv-only", "ECA-only", "Hybrid"] if v in eff.index]

    vals_p = [float(eff.loc[v, "params_M"]) / base_p * 100.0 - 100.0 for v in variants]
    vals_f = [float(eff.loc[v, "flops_M"]) / base_f * 100.0 - 100.0 for v in variants]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Params",
            y=variants,
            x=vals_p,
            orientation="h",
            marker_color=[cmap.get(v, "#999999") for v in variants],
        )
    )
    fig.add_trace(
        go.Bar(
            name="FLOPs",
            y=variants,
            x=vals_f,
            orientation="h",
            marker_color=[cmap.get(v, "#999999") for v in variants],
            opacity=0.65,
        )
    )

    fig.update_layout(
        barmode="group",
        template=plotly_template(theme),  # type: ignore[arg-type]
        title="Parameter and FLOP Overhead Relative to Baseline MobileNetV2 (±10% budget)",
        xaxis_title="Overhead vs Baseline (%)",
        yaxis_title="Variant",
        **plotly_font_kwargs(),
        height=560,
        legend_title_text="Metric",
    )
    # reference lines
    for x in (-10, 10):
        fig.add_vline(x=x, line_width=1, line_dash="dash", line_color="#666666")
    fig.add_vline(x=0, line_width=1, line_dash="solid", line_color="#333333")
    return fig

