from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from chapter4_dashboard.utils.colors import plotly_font_kwargs, plotly_template
from chapter4_dashboard.utils.constants import VARIANT_ORDER


def build_efficiency_radar(
    df_runs: pd.DataFrame,
    df_eff: pd.DataFrame,
    cmap: dict[str, str],
    theme: str,
    normalized: bool,
) -> go.Figure:
    if df_eff.empty:
        return go.Figure()
    eff = df_eff.set_index("variant")
    if "Baseline" not in eff.index:
        return go.Figure()
    base = eff.loc["Baseline"]

    # Use CIFAR-100 top1 for accuracy axis.
    acc = (
        df_runs[(df_runs["dataset"] == "CIFAR-100")]
        .groupby("variant")["top1_acc"]
        .mean()
        .to_dict()
    )
    base_acc = float(acc.get("Baseline", np.nan))

    axes = ["Params (M)", "FLOPs (M)", "Size (MB)", "Latency (ms)", "Top-1 (CIFAR-100)"]
    fig = go.Figure()
    for v in VARIANT_ORDER:
        if v not in eff.index:
            continue
        vals = [
            float(eff.loc[v, "params_M"]),
            float(eff.loc[v, "flops_M"]),
            float(eff.loc[v, "size_mb"]),
            float(eff.loc[v, "latency_ms"]),
            float(acc.get(v, np.nan)),
        ]
        if normalized:
            vals = [
                vals[0] / float(base["params_M"]),
                vals[1] / float(base["flops_M"]),
                vals[2] / float(base["size_mb"]),
                vals[3] / float(base["latency_ms"]),
                (2.0 - (vals[4] / base_acc)) if np.isfinite(base_acc) and base_acc > 0 else np.nan,
            ]

        # close the polygon
        fig.add_trace(
            go.Scatterpolar(
                r=vals + [vals[0]],
                theta=axes + [axes[0]],
                fill="toself",
                name=v,
                opacity=0.25,
                line=dict(color=cmap.get(v, "#999999"), width=2),
            )
        )

    fig.update_layout(
        template=plotly_template(theme),  # type: ignore[arg-type]
        title="Efficiency radar" + (" (Normalized ÷ Baseline)" if normalized else " (Absolute)"),
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        height=520,
        **plotly_font_kwargs(),
    )
    return fig

