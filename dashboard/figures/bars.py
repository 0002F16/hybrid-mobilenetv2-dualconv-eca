"""Tab 4: cross-dataset grouped bar charts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dashboard.constants import DATASET_ORDER
from dashboard.stats.tests import bootstrap_mean_ci_across_seeds
from dashboard.utils.colors import axis_tick_font, plotly_font_kwargs, plotly_template


def build_cross_dataset_bars(
    df_runs: pd.DataFrame,
    datasets: list[str],
    variants: list[str],
    metric: str,
    color_map: dict[str, str],
    theme: str,
    show_ci: bool,
    delta_mode: bool,
) -> go.Figure:
    col = "top1_acc" if metric == "top1" else "top5_acc"
    tpl = plotly_template(theme)  # type: ignore[arg-type]
    ds_list = [d for d in DATASET_ORDER if d in datasets]

    fig = go.Figure()
    for v in variants:
        ys = []
        err_plus = []
        err_minus = []
        for d in ds_list:
            v_vals = df_runs[(df_runs["dataset"] == d) & (df_runs["variant"] == v)][col].values
            b_vals = df_runs[(df_runs["dataset"] == d) & (df_runs["variant"] == "Baseline")][
                col
            ].values
            if delta_mode:
                paired = []
                for s in sorted(set(df_runs["seed"].unique())):
                    vv = df_runs[
                        (df_runs["dataset"] == d)
                        & (df_runs["variant"] == v)
                        & (df_runs["seed"] == s)
                    ][col].values
                    bb = df_runs[
                        (df_runs["dataset"] == d)
                        & (df_runs["variant"] == "Baseline")
                        & (df_runs["seed"] == s)
                    ][col].values
                    if len(vv) and len(bb):
                        paired.append(float(vv[0] - bb[0]))
                if not paired:
                    ys.append(0)
                    err_plus.append(0)
                    err_minus.append(0)
                    continue
                arr = np.asarray(paired)
                m = float(np.mean(arr))
                ys.append(m)
                if show_ci:
                    lo, hi = bootstrap_mean_ci_across_seeds(arr)
                    err_minus.append(max(0, m - lo))
                    err_plus.append(max(0, hi - m))
                else:
                    s = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
                    err_plus.append(s)
                    err_minus.append(s)
            else:
                if len(v_vals) == 0:
                    ys.append(0)
                    err_plus.append(0)
                    err_minus.append(0)
                    continue
                m = float(np.mean(v_vals))
                ys.append(m)
                if show_ci:
                    lo, hi = bootstrap_mean_ci_across_seeds(v_vals)
                    err_minus.append(max(0, m - lo))
                    err_plus.append(max(0, hi - m))
                else:
                    s = float(np.std(v_vals, ddof=1)) if len(v_vals) > 1 else 0.0
                    err_plus.append(s)
                    err_minus.append(s)

        marker_color = color_map.get(v, "#888888")
        if delta_mode:
            marker_color = [
                "green" if y >= 0 else "crimson" if y < 0 else marker_color for y in ys
            ]
        fig.add_trace(
            go.Bar(
                name=v,
                x=ds_list,
                y=ys,
                marker_color=marker_color,
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=err_plus,
                    arrayminus=err_minus,
                    color="rgba(0,0,0,0.35)",
                    thickness=1.5,
                    width=4,
                ),
            )
        )

    if delta_mode:
        fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)

    fig.update_layout(
        barmode="group",
        template=tpl,
        **plotly_font_kwargs(),
        bargap=0.15,
        xaxis=dict(title="Dataset", **axis_tick_font()),
        yaxis=dict(
            title=f"{'Δ ' if delta_mode else ''}{metric.upper()} (%)",
            **axis_tick_font(),
        ),
        height=480,
        width=1000,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig
