"""Tab 7: training curves."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dashboard.constants import VARIANT_ORDER
from dashboard.utils.colors import axis_tick_font, plotly_font_kwargs, plotly_template


def _smooth(s: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return s
    return s.rolling(window=window, min_periods=1, center=True).mean()


def build_training_curves(
    df_curves: pd.DataFrame,
    dataset: str,
    seed: int | None,
    all_seeds: bool,
    metric: str,
    window: int,
    color_map: dict[str, str],
    theme: str,
) -> go.Figure:
    tpl = plotly_template(theme)  # type: ignore[arg-type]
    col_map = {
        "Val Top-1": "val_top1",
        "Train Loss": "train_loss",
        "Val Loss": "val_loss",
    }
    ycol = col_map[metric]
    fig = go.Figure()

    for v in VARIANT_ORDER:
        sub = df_curves[(df_curves["dataset"] == dataset) & (df_curves["variant"] == v)]
        if sub.empty:
            continue
        c = color_map.get(v, "#888")
        if not all_seeds and seed is not None:
            ssub = sub[sub["seed"] == seed].sort_values("epoch")
            if ssub.empty:
                continue
            y = _smooth(ssub[ycol], window)
            fig.add_trace(
                go.Scatter(
                    x=ssub["epoch"],
                    y=y,
                    mode="lines",
                    name=v,
                    line=dict(color=c, width=2),
                )
            )
            best_ep = int(ssub.loc[ssub["val_top1"].idxmax(), "epoch"])
            fig.add_vline(x=best_ep, line_dash="dash", line_color=c, opacity=0.6)
        else:
            epochs = sorted(sub["epoch"].unique())
            means = []
            stds = []
            for e in epochs:
                chunk = sub[sub["epoch"] == e][ycol]
                means.append(float(chunk.mean()))
                stds.append(float(chunk.std(ddof=1)) if len(chunk) > 1 else 0.0)
            m_series = pd.Series(means)
            m_smooth = _smooth(m_series, window)
            best_eps = []
            for sd in sub["seed"].unique():
                ssub = sub[sub["seed"] == sd]
                if len(ssub):
                    best_eps.append(int(ssub.loc[ssub["val_top1"].idxmax(), "epoch"]))
            mean_best_ep = int(np.round(np.mean(best_eps))) if best_eps else 1
            fig.add_vline(x=mean_best_ep, line_dash="dash", line_color=c, opacity=0.5)
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=m_smooth,
                    mode="lines",
                    name=v,
                    line=dict(color=c, width=3),
                )
            )
            upper = m_smooth + np.asarray(stds)
            lower = m_smooth - np.asarray(stds)
            fig.add_trace(
                go.Scatter(
                    x=epochs + epochs[::-1],
                    y=list(upper) + list(lower[::-1]),
                    fill="toself",
                    fillcolor=c,
                    opacity=0.15,
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                    name=f"{v} band",
                )
            )

    fig.update_layout(
        template=tpl,
        **plotly_font_kwargs(),
        xaxis=dict(title="Epoch", **axis_tick_font()),
        yaxis=dict(title=metric, **axis_tick_font()),
        height=520,
        width=1200,
        legend=dict(orientation="h", y=1.05),
    )
    return fig


def convergence_summary(df_curves: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows = []
    base_best = None
    for v in VARIANT_ORDER:
        sub = df_curves[(df_curves["dataset"] == dataset) & (df_curves["variant"] == v)]
        if sub.empty:
            continue
        best_vals = []
        epochs = []
        final_train = []
        for s in sub["seed"].unique():
            ss = sub[sub["seed"] == s].sort_values("epoch")
            if len(ss) == 0:
                continue
            im = ss["val_top1"].idxmax()
            r = ss.loc[im]
            best_vals.append(float(r["val_top1"]))
            epochs.append(int(r["epoch"]))
            final_train.append(float(ss["train_loss"].iloc[-1]))
        best_val = float(np.mean(best_vals)) if best_vals else float("nan")
        at_ep = int(round(np.mean(epochs))) if epochs else 0
        fl = float(np.mean(final_train)) if final_train else float("nan")
        row = {
            "Variant": v,
            "Best Val Top-1": best_val,
            "At Epoch": at_ep,
            "Final Train Loss": float(fl),
        }
        if "curves_source" in sub.columns:
            srcs = sub.groupby("seed")["curves_source"].first()
            row["Curves source"] = "+".join(sorted(srcs.unique()))
        rows.append(row)
        if v == "Baseline":
            base_best = best_val
    out = pd.DataFrame(rows)
    if base_best is not None:
        out["Δ Best vs Baseline (pp)"] = out["Best Val Top-1"] - base_best
    return out
