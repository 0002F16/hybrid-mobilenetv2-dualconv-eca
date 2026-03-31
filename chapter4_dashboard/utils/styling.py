from __future__ import annotations

import pandas as pd


def _white_to_green_css(value: float, vmin: float, vmax: float) -> str:
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
    df = df.reset_index(drop=True)
    styler = df.style.format({"top1_acc": "{:.2f}", "top5_acc": "{:.2f}"})

    def gradient_and_bold(data: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame("", index=data.index, columns=data.columns)
        if "dataset" not in data.columns:
            return out
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
                    css = _white_to_green_css(float(data.loc[i, col]), vmin, vmax)
                    if i == mi:
                        css += "; font-weight: bold"
                    out.loc[i, col] = css
        return out

    return styler.apply(gradient_and_bold, axis=None)


def style_efficiency_delta(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def color_delta(val):
        if pd.isna(val):
            return ""
        v = abs(float(val))
        return "background-color: #d4edda" if v <= 10 else "background-color: #f8d7da"

    fmt = {
        "params_M": "{:.2f}",
        "flops_M": "{:.1f}",
        "size_mb": "{:.2f}",
        "latency_ms": "{:.2f}",
        "delta_params_pct": "{:+.2f}",
        "delta_flops_pct": "{:+.2f}",
    }
    sty = df.style.format({k: v for k, v in fmt.items() if k in df.columns})
    subset = [c for c in ["delta_params_pct", "delta_flops_pct"] if c in df.columns]
    if subset:
        try:
            sty = sty.map(color_delta, subset=subset)
        except AttributeError:
            sty = sty.applymap(color_delta, subset=subset)
    return sty


def style_budget_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def color_pct(val):
        if pd.isna(val):
            return ""
        v = abs(float(val))
        return "background-color: #d4edda" if v <= 10 else "background-color: #f8d7da"

    def baseline_row(row):
        if str(row.get("Variant", "")) == "Baseline":
            return ["background-color: #e9ecef"] * len(row)
        return [""] * len(row)

    sty = df.style.apply(baseline_row, axis=1)
    for col in ["Δ Params %", "Δ FLOPs %"]:
        if col in df.columns:
            try:
                sty = sty.map(color_pct, subset=[col])
            except AttributeError:
                sty = sty.applymap(color_pct, subset=[col])
    return sty

