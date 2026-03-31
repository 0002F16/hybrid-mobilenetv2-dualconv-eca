from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from chapter4_dashboard.utils.colors import plotly_font_kwargs, plotly_template
from chapter4_dashboard.utils.constants import SEEDS, VARIANT_ORDER


def _smooth(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1, center=False).mean()


def build_training_diagnostic_plot(
    df_curves: pd.DataFrame,
    dataset: str,
    metric: str,
    view_mode: str,
    seed: int | None,
    smoothing: int,
    cmap: dict[str, str],
    theme: str,
) -> go.Figure:
    if df_curves.empty:
        return go.Figure()
    fig = go.Figure()

    metric_col = {"Val Top-1": "val_top1", "Train Loss": "train_loss", "Val Loss": "val_loss"}[metric]

    for v in VARIANT_ORDER:
        sub = df_curves[(df_curves["dataset"] == dataset) & (df_curves["variant"] == v)]
        if sub.empty:
            continue
        if view_mode == "Single seed" and seed is not None:
            sub = sub[sub["seed"] == seed]
            g = sub.sort_values("epoch")
            y = _smooth(g[metric_col].astype(float), smoothing)
            fig.add_trace(
                go.Scatter(
                    x=g["epoch"],
                    y=y,
                    mode="lines",
                    name=v,
                    line=dict(color=cmap.get(v, "#999999"), width=2),
                )
            )
            if "is_best_epoch" in g.columns and metric == "Val Top-1":
                best = g[g["is_best_epoch"] == True]
                if len(best):
                    ep = int(best["epoch"].iloc[0])
                    top = float(best["val_top1"].iloc[0])
                    fig.add_vline(x=ep, line_width=1, line_dash="dot", line_color=cmap.get(v, "#999999"))
                    fig.add_annotation(
                        x=ep,
                        y=top,
                        text=f"Best: {top:.2f}% @ ep {ep}",
                        showarrow=True,
                        arrowhead=2,
                        ax=20,
                        ay=-20,
                        font=dict(size=10),
                    )
        else:
            # all seeds mean ± sd band
            g = sub.groupby("epoch", as_index=False)[metric_col].agg(["mean", "std"])
            g.columns = ["epoch", "mean", "std"]
            g = g.sort_values("epoch")
            y = _smooth(g["mean"].astype(float), smoothing)
            sd = _smooth(g["std"].fillna(0.0).astype(float), smoothing)
            color = cmap.get(v, "#999999")
            fig.add_trace(
                go.Scatter(
                    x=g["epoch"],
                    y=y,
                    mode="lines",
                    name=v,
                    line=dict(color=color, width=3),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=list(g["epoch"]) + list(g["epoch"][::-1]),
                    y=list((y + sd)) + list((y - sd)[::-1]),
                    fill="toself",
                    fillcolor=_rgba(color, 0.2),
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    fig.update_layout(
        template=plotly_template(theme),  # type: ignore[arg-type]
        height=560,
        title=f"{dataset} — {metric}",
        xaxis_title="Epoch",
        yaxis_title=metric,
        **plotly_font_kwargs(),
    )
    return fig


def convergence_summary_table(df_curves: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if df_curves.empty:
        return pd.DataFrame()
    sub = df_curves[df_curves["dataset"] == dataset].copy()
    if sub.empty:
        return pd.DataFrame()

    # best per seed, variant
    if "is_best_epoch" in sub.columns:
        best_rows = sub[sub["is_best_epoch"] == True]
    else:
        best_rows = sub.sort_values("val_top1").groupby(["variant", "seed"], as_index=False).tail(1)

    agg = best_rows.groupby("variant", as_index=False).agg(
        best_val_top1_mean=("val_top1", "mean"),
        best_val_top1_std=("val_top1", "std"),
        best_epoch_mean=("epoch", "mean"),
    )

    # epochs to 95% of best (approx using mean curve)
    out = []
    base = agg[agg["variant"] == "Baseline"]
    base_best = float(base["best_val_top1_mean"].iloc[0]) if len(base) else float("nan")
    for _, r in agg.iterrows():
        v = r["variant"]
        target = 0.95 * float(r["best_val_top1_mean"])
        mean_curve = sub[sub["variant"] == v].groupby("epoch")["val_top1"].mean()
        hit = mean_curve[mean_curve >= target]
        ep95 = int(hit.index.min()) if len(hit) else int(r["best_epoch_mean"])
        out.append(
            {
                "Variant": v,
                "Best Val Top-1 (mean±std)": f"{r['best_val_top1_mean']:.2f} ± {r['best_val_top1_std']:.2f}",
                "Best Epoch (mean)": float(r["best_epoch_mean"]),
                "Δ vs Baseline (pp)": float(r["best_val_top1_mean"]) - base_best if np.isfinite(base_best) else np.nan,
                "Epochs to 95% of Best": ep95,
            }
        )
    return pd.DataFrame(out)


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

