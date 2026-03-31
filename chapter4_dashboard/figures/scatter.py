from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from chapter4_dashboard.utils.colors import plotly_font_kwargs, plotly_template
from chapter4_dashboard.utils.constants import DATASET_ORDER, VARIANT_ORDER


def build_accuracy_efficiency_scatter(
    df_runs: pd.DataFrame, df_eff: pd.DataFrame, cmap: dict[str, str], theme: str
) -> go.Figure:
    if df_runs.empty or df_eff.empty:
        return go.Figure()
    eff = df_eff.set_index("variant")
    base = eff.loc["Baseline"] if "Baseline" in eff.index else None

    # mean & std top1 per dataset/variant
    agg = (
        df_runs.groupby(["dataset", "variant"], as_index=False)
        .agg(top1_mean=("top1_acc", "mean"), top1_std=("top1_acc", "std"))
    )

    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, subplot_titles=DATASET_ORDER)
    ymins, ymaxs = [], []
    for col_i, ds in enumerate(DATASET_ORDER, start=1):
        sub = agg[agg["dataset"] == ds]
        for v in VARIANT_ORDER:
            if v not in eff.index:
                continue
            row = sub[sub["variant"] == v]
            if row.empty:
                continue
            mu = float(row["top1_mean"].iloc[0])
            sd = float(row["top1_std"].iloc[0]) if np.isfinite(row["top1_std"].iloc[0]) else 0.0
            x = float(eff.loc[v, "flops_M"])
            size = float(eff.loc[v, "size_mb"])
            ymins.append(mu - sd)
            ymaxs.append(mu + sd)
            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[mu],
                    mode="markers+text",
                    text=[v],
                    textposition="top center",
                    marker=dict(size=max(10, 6 + size * 1.2), color=cmap.get(v, "#999999")),
                    error_y=dict(type="data", array=[sd], visible=True),
                    showlegend=False,
                ),
                row=1,
                col=col_i,
            )
        if base is not None:
            bx = float(base["flops_M"])
            fig.add_vline(x=bx * 0.9, line_width=1, line_dash="dash", line_color="#888888", row=1, col=col_i)
            fig.add_vline(x=bx * 1.1, line_width=1, line_dash="dash", line_color="#888888", row=1, col=col_i)
            bmu = float(agg[(agg["dataset"] == ds) & (agg["variant"] == "Baseline")]["top1_mean"].iloc[0])
            fig.add_hline(y=bmu, line_width=1, line_dash="dash", line_color="#888888", row=1, col=col_i)

    if ymins and ymaxs:
        pad = 0.6
        fig.update_yaxes(range=[min(ymins) - pad, max(ymaxs) + pad])

    fig.update_layout(
        template=plotly_template(theme),  # type: ignore[arg-type]
        height=520,
        title="Accuracy–Efficiency Trade-off (Top-1 vs FLOPs)",
        **plotly_font_kwargs(),
    )
    fig.update_xaxes(title_text="FLOPs (M)")
    fig.update_yaxes(title_text="Top-1 mean (%)")
    return fig

