from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from chapter4_dashboard.figures.bars import build_accuracy_bars
from chapter4_dashboard.utils.captions import caption_table_main_results, caption_table_stats
from chapter4_dashboard.utils.export import dataframe_to_csv_bytes, figure_to_png_bytes


def _main_results_table(df_runs: pd.DataFrame, df_eff: pd.DataFrame) -> pd.DataFrame:
    if df_runs.empty or df_eff.empty or "Baseline" not in df_eff["variant"].values:
        return pd.DataFrame()
    eff = df_eff.set_index("variant")
    base_p = float(eff.loc["Baseline", "params_M"])
    base_f = float(eff.loc["Baseline", "flops_M"])

    agg = (
        df_runs.groupby(["dataset", "variant"], as_index=False)
        .agg(
            top1_mean=("top1_acc", "mean"),
            top1_std=("top1_acc", "std"),
            top5_mean=("top5_acc", "mean"),
            top5_std=("top5_acc", "std"),
        )
    )
    rows = []
    for _, r in agg.iterrows():
        v = r["variant"]
        ds = r["dataset"]
        if v not in eff.index:
            continue
        dpp = (float(eff.loc[v, "params_M"]) - base_p) / base_p * 100.0
        dff = (float(eff.loc[v, "flops_M"]) - base_f) / base_f * 100.0
        rows.append(
            {
                "Dataset": ds,
                "Variant": v,
                "Top-1 (mean±std)": f"{r['top1_mean']:.2f} ± {r['top1_std']:.2f}",
                "Top-5 (mean±std)": f"{r['top5_mean']:.2f} ± {r['top5_std']:.2f}",
                "Params(M)": float(eff.loc[v, "params_M"]),
                "ΔParams%": dpp,
                "FLOPs(M)": float(eff.loc[v, "flops_M"]),
                "ΔFLOPs%": dff,
                "Size(MB)": float(eff.loc[v, "size_mb"]),
                "Latency(ms)": float(eff.loc[v, "latency_ms"]),
                "_top1_mean": float(r["top1_mean"]),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # rank within dataset by top1 mean (hidden helper column)
    df["rank"] = df.groupby("Dataset")["_top1_mean"].rank(ascending=False, method="min")
    return df


def render_tab_performance(df_runs: pd.DataFrame, df_eff: pd.DataFrame, cmap: dict[str, str]) -> None:
    st.subheader("Tab 3 — Objective 2: Performance Results")
    st.info(
        "“Specific Objective 2: Demonstrate that the proposed MobileNetV2 architectural variants "
        "achieve improved classification performance across CIFAR-10, CIFAR-100, and Tiny-ImageNet.”"
    )

    st.markdown("**3A — Main Results Table**")
    tbl = _main_results_table(df_runs, df_eff)
    if tbl.empty:
        st.warning("Main results table requires `df_runs` and `df_efficiency` with a Baseline row.")
    else:
        def style(df: pd.DataFrame) -> pd.io.formats.style.Styler:
            sty = df.drop(columns=["_top1_mean"]).style

            def hi(data):
                out = pd.DataFrame("", index=data.index, columns=data.columns)
                for ds in data["Dataset"].unique():
                    sub = data[data["Dataset"] == ds]
                    best = sub.index[sub["rank"] == 1]
                    second = sub.index[sub["rank"] == 2]
                    out.loc[best, :] = "background-color: #c8e6c9; font-weight: bold"
                    out.loc[second, :] = "background-color: #e8f5e9"
                    base = sub.index[sub["Variant"] == "Baseline"]
                    out.loc[base, :] = "background-color: #e9ecef"
                return out

            sty = sty.apply(hi, axis=None)
            for c in ["ΔParams%", "ΔFLOPs%"]:
                if c in df.columns:
                    try:
                        sty = sty.map(lambda v: "background-color: #d4edda" if abs(float(v)) <= 10 else "background-color: #f8d7da", subset=[c])
                    except AttributeError:
                        sty = sty.applymap(lambda v: "background-color: #d4edda" if abs(float(v)) <= 10 else "background-color: #f8d7da", subset=[c])
            return sty.format(
                {
                    "Params(M)": "{:.2f}",
                    "ΔParams%": "{:+.2f}",
                    "FLOPs(M)": "{:.1f}",
                    "ΔFLOPs%": "{:+.2f}",
                    "Size(MB)": "{:.2f}",
                    "Latency(ms)": "{:.2f}",
                }
            )

        st.dataframe(style(tbl), use_container_width=True, height=420)
        st.download_button(
            "Download Main results table (CSV)",
            dataframe_to_csv_bytes(tbl.drop(columns=["_top1_mean"])),
            file_name="main_results_table.csv",
            mime="text/csv",
        )
        st.code(caption_table_main_results())

    st.markdown("**3B — Accuracy Bar Charts**")
    mode = st.radio("Mode", ["Absolute accuracy", "Δ vs Baseline"], horizontal=True, key="perf_mode")
    delta_mode = "Δ" in mode
    fig1 = build_accuracy_bars(df_runs, "top1", cmap, st.session_state.theme, st.session_state.show_ci, delta_mode)
    st.plotly_chart(fig1, use_container_width=True)
    st.download_button(
        "Download Top-1 bars (PNG)",
        figure_to_png_bytes(fig1),
        file_name="top1_bars.png",
        mime="image/png",
    )
    fig2 = build_accuracy_bars(df_runs, "top5", cmap, st.session_state.theme, st.session_state.show_ci, delta_mode)
    st.plotly_chart(fig2, use_container_width=True)
    st.download_button(
        "Download Top-5 bars (PNG)",
        figure_to_png_bytes(fig2),
        file_name="top5_bars.png",
        mime="image/png",
    )

    st.markdown("**3C — Statistical Validation Table**")
    st.warning(
        "With n=5 seeds, Wilcoxon signed-rank has limited statistical power. Corrected p-values and effect sizes "
        "(median Δ, CI) are the primary evidence. Significance stars are supplementary."
    )
    tb = st.session_state.get("stats_results")
    if tb is None or (hasattr(tb, "empty") and tb.empty):
        st.info("No stats available (need df_runs with Baseline and variant rows per dataset/seed).")
    else:
        st.dataframe(
            tb.style.format(
                {"Median Δ (pp)": "{:+.3f}", "W stat": "{:.0f}", "Raw p": "{:.4f}", "Corrected p": "{:.4f}"}
            ),
            use_container_width=True,
            height=360,
        )
        st.download_button(
            "Download Table B (CSV)",
            dataframe_to_csv_bytes(tb),
            file_name="table_b_stats.csv",
            mime="text/csv",
        )
        st.code(caption_table_stats())

