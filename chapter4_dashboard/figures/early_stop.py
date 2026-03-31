from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from chapter4_dashboard.utils.colors import plotly_font_kwargs, plotly_template
from chapter4_dashboard.utils.constants import VARIANT_ORDER


def build_early_stopping_distribution(df_curves: pd.DataFrame, dataset: str, cmap: dict[str, str], theme: str) -> go.Figure:
    if df_curves.empty:
        return go.Figure()
    sub = df_curves[df_curves["dataset"] == dataset].copy()
    if sub.empty or "is_best_epoch" not in sub.columns:
        return go.Figure()

    best = sub[sub["is_best_epoch"] == True][["variant", "seed", "epoch"]]
    fig = go.Figure()
    for v in VARIANT_ORDER:
        vals = best[best["variant"] == v]["epoch"].astype(float)
        if vals.empty:
            continue
        fig.add_trace(
            go.Violin(
                y=vals,
                x=[v] * len(vals),
                name=v,
                box_visible=True,
                meanline_visible=True,
                line_color=cmap.get(v, "#999999"),
                fillcolor=cmap.get(v, "#999999"),
                opacity=0.35,
                showlegend=False,
            )
        )

    fig.update_layout(
        template=plotly_template(theme),  # type: ignore[arg-type]
        title="Early stopping distribution (best_epoch)",
        xaxis_title="Variant",
        yaxis_title="Best epoch",
        height=420,
        **plotly_font_kwargs(),
    )
    return fig

