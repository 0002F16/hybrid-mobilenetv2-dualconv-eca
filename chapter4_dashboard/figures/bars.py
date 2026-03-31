from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from chapter4_dashboard.utils.colors import plotly_font_kwargs, plotly_template
from chapter4_dashboard.utils.constants import DATASET_ORDER, VARIANT_ORDER


def _bootstrap_ci_mean(x: np.ndarray, n: int = 1000, alpha: float = 0.05, seed: int = 123) -> tuple[float, float]:
    g = np.random.default_rng(seed)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    means = []
    for _ in range(n):
        sample = g.choice(x, size=len(x), replace=True)
        means.append(float(np.mean(sample)))
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def build_accuracy_bars(
    df_runs: pd.DataFrame,
    metric: str,
    cmap: dict[str, str],
    theme: str,
    show_ci: bool,
    delta_mode: bool,
) -> go.Figure:
    col = "top1_acc" if metric == "top1" else "top5_acc"
    if df_runs.empty:
        return go.Figure()

    # Compute baseline means per dataset for delta mode.
    base_means = (
        df_runs[df_runs["variant"] == "Baseline"]
        .groupby("dataset")[col]
        .mean()
        .to_dict()
    )

    fig = go.Figure()
    for v in VARIANT_ORDER:
        sub = df_runs[df_runs["variant"] == v]
        xs, ys, errs = [], [], []
        ci_l, ci_u = [], []
        for ds in DATASET_ORDER:
            vals = sub[sub["dataset"] == ds][col].astype(float).to_numpy()
            mu = float(np.mean(vals)) if len(vals) else float("nan")
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            if delta_mode:
                mu = mu - float(base_means.get(ds, 0.0))
            xs.append(ds)
            ys.append(mu)
            errs.append(sd)
            if show_ci and len(vals):
                lo, hi = _bootstrap_ci_mean(vals)
                if delta_mode:
                    lo -= float(base_means.get(ds, 0.0))
                    hi -= float(base_means.get(ds, 0.0))
                ci_l.append(lo)
                ci_u.append(hi)
            else:
                ci_l.append(float("nan"))
                ci_u.append(float("nan"))

        marker_color = cmap.get(v, "#999999")
        if delta_mode and v != "Baseline":
            # color by sign via per-point colors
            colors = ["#2ca02c" if y > 0 else ("#d62728" if y < 0 else "#999999") for y in ys]
        else:
            colors = [marker_color] * len(xs)

        fig.add_trace(
            go.Bar(
                name=v,
                x=xs,
                y=ys,
                marker_color=colors,
                error_y=dict(type="data", array=errs, visible=True),
            )
        )

        if show_ci:
            fig.add_trace(
                go.Scatter(
                    name=f"{v} 95% CI",
                    x=xs,
                    y=[(l + u) / 2 for l, u in zip(ci_l, ci_u)],
                    mode="markers",
                    marker=dict(color=marker_color, size=6, symbol="line-ns-open"),
                    error_y=dict(
                        type="data",
                        array=[(u - l) / 2 if np.isfinite(l) and np.isfinite(u) else 0 for l, u in zip(ci_l, ci_u)],
                        visible=True,
                        thickness=1,
                        width=0,
                    ),
                    showlegend=False,
                )
            )

    title = ("Top-1" if metric == "top1" else "Top-5") + (" (Δ vs Baseline)" if delta_mode else "")
    fig.update_layout(
        template=plotly_template(theme),  # type: ignore[arg-type]
        barmode="group",
        title=title,
        xaxis_title="Dataset",
        yaxis_title="Δ Accuracy (pp)" if delta_mode else "Accuracy (%)",
        **plotly_font_kwargs(),
        height=520,
    )
    if delta_mode:
        fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="#666666")
    return fig

