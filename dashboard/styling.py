"""Pandas Styler helpers for Tab 1 and Tab 2."""

from __future__ import annotations

import pandas as pd


def _white_to_green_css(value: float, vmin: float, vmax: float) -> str:
    """Inline RGB background white → green (no matplotlib)."""
    if pd.isna(value):
        return ""
    span = (vmax - vmin) if vmax > vmin else 1.0
    t = (float(value) - vmin) / span
    t = max(0.0, min(1.0, t))
    r = int(255 - t * 95)
    g = int(255 - t * 35)
    b = int(255 - t * 95)
    return f"background-color: rgb({r},{g},{b})"


def style_runs_raw(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Gradient per dataset on top1/top5; bold max per dataset per metric."""
    df = df.reset_index(drop=True)
    fmt_cols = {c: "{:.4f}" for c in ["top1_acc", "top5_acc"] if c in df.columns}
    styler = df.style.format(fmt_cols)

    def gradient_and_bold(data: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame("", index=data.index, columns=data.columns)
        for ds in data["dataset"].unique():
            sub = data[data["dataset"] == ds]
            idx = sub.index.tolist()
            for col in ["top1_acc", "top5_acc"]:
                if col not in data.columns:
                    continue
                series = sub[col].astype(float)
                vmin, vmax = float(series.min()), float(series.max())
                mi = series.idxmax()
                for i in idx:
                    v = float(data.loc[i, col])
                    css = _white_to_green_css(v, vmin, vmax)
                    if i == mi:
                        css += "; font-weight: bold"
                    out.loc[i, col] = css
        return out

    styler = styler.apply(gradient_and_bold, axis=None)
    return styler


def style_main_results_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Best Top-1 per dataset: bold + green row; ΔParams/ΔFLOPs magnitude; Δacc vs baseline signed."""

    def row_highlight(data: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame("", index=data.index, columns=data.columns)
        if "Dataset" in data.columns:
            for ds in data["Dataset"].unique():
                sub = data[data["Dataset"] == ds]
                i = sub["top1_mean"].idxmax()
                out.loc[i, :] = "background-color: #c8e6c9; font-weight: bold"
        else:
            i = data["top1_mean"].idxmax()
            out.loc[i, :] = "background-color: #c8e6c9; font-weight: bold"
        return out

    def color_delta_eff(val):
        if pd.isna(val):
            return ""
        v = abs(float(val))
        if v <= 5:
            return "background-color: #d4edda"
        if v <= 10:
            return "background-color: #fff3cd"
        return "background-color: #f8d7da"

    def color_delta_acc(val):
        if pd.isna(val):
            return ""
        v = float(val)
        if v > 1e-6:
            return "background-color: #d4edda"
        if v < -1e-6:
            return "background-color: #f8d7da"
        return "background-color: #e9ecef"

    sty = df.style.apply(row_highlight, axis=None)
    acc_cols = [c for c in ["ΔTop-1 pp", "ΔTop-5 pp"] if c in df.columns]
    eff_cols = [c for c in ["ΔParams%", "ΔFLOPs%"] if c in df.columns]
    try:
        if eff_cols:
            sty = sty.map(color_delta_eff, subset=eff_cols)
        if acc_cols:
            sty = sty.map(color_delta_acc, subset=acc_cols)
    except AttributeError:
        if eff_cols:
            sty = sty.applymap(color_delta_eff, subset=eff_cols)
        if acc_cols:
            sty = sty.applymap(color_delta_acc, subset=acc_cols)
    return sty


def style_efficiency_delta(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Green if |delta|<=10%, red if >10%."""

    def color_delta(val):
        if pd.isna(val):
            return ""
        v = abs(float(val))
        if v <= 10:
            return "background-color: #d4edda"
        return "background-color: #f8d7da"

    if "delta_params_pct" in df.columns:
        try:
            sty = df.style.map(color_delta, subset=["delta_params_pct", "delta_flops_pct"])
        except AttributeError:
            sty = df.style.applymap(color_delta, subset=["delta_params_pct", "delta_flops_pct"])
    else:
        sty = df.style
    return sty
