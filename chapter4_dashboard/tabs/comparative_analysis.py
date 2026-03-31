from __future__ import annotations

import pandas as pd
import streamlit as st

from chapter4_dashboard.figures.radar import build_efficiency_radar
from chapter4_dashboard.figures.scatter import build_accuracy_efficiency_scatter
from chapter4_dashboard.utils.export import dataframe_to_csv_bytes, figure_to_png_bytes


def _default_literature_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Study": "Author et al. (Year)",
                "Backbone": "MobileNetV2",
                "Modification": "—",
                "Dataset": "CIFAR-100",
                "Top-1": 0.0,
                "Params(M)": 0.0,
                "FLOPs(M)": 0.0,
                "Compared To This Study": "",
            }
        ]
    )


def _autofill_compared(df_lit: pd.DataFrame, df_runs: pd.DataFrame) -> pd.DataFrame:
    out = df_lit.copy()
    if df_runs.empty or "Hybrid" not in df_runs["variant"].unique():
        return out
    hyb = (
        df_runs[df_runs["variant"] == "Hybrid"]
        .groupby("dataset")["top1_acc"]
        .mean()
        .to_dict()
    )
    for i, r in out.iterrows():
        ds = str(r.get("Dataset", ""))
        if ds in hyb and pd.notna(r.get("Top-1", None)):
            theirs = float(r.get("Top-1", 0.0))
            ours = float(hyb[ds])
            out.at[i, "Compared To This Study"] = f"This study (Hybrid): {ours:.2f}% ({ours-theirs:+.2f} pp vs their result)"
    return out


def render_tab_comparative(df_runs: pd.DataFrame, df_eff: pd.DataFrame, cmap: dict[str, str]) -> None:
    st.subheader("Tab 5 — 4.2 Comparative Analysis")
    st.caption("Compare proposed model against baselines quantitatively and against prior literature.")

    st.markdown("**5A — Accuracy–Efficiency Scatter**")
    fig_s = build_accuracy_efficiency_scatter(df_runs, df_eff, cmap, st.session_state.theme)
    st.plotly_chart(fig_s, use_container_width=True)
    st.download_button(
        "Download scatter (PNG)",
        figure_to_png_bytes(fig_s),
        file_name="comparative_scatter.png",
        mime="image/png",
    )

    st.markdown("**5B — Efficiency Radar**")
    norm = st.radio(
        "Radar mode",
        ["Normalized (÷ Baseline)", "Absolute values"],
        horizontal=True,
        key="radar_mode",
    )
    fig_r = build_efficiency_radar(
        df_runs,
        df_eff,
        cmap,
        st.session_state.theme,
        normalized=("Normalized" in norm),
    )
    st.plotly_chart(fig_r, use_container_width=True)
    st.download_button(
        "Download radar (PNG)",
        figure_to_png_bytes(fig_r),
        file_name="efficiency_radar.png",
        mime="image/png",
    )
    st.caption("All axes normalized to Baseline = 1.0. For accuracy axis, outward = better.")

    st.markdown("**5C — Literature Comparison Table**")
    st.info(
        "Populate this table manually from your reviewed literature. Reported this-study values are auto-filled "
        "from uploaded experimental results."
    )
    if "lit_table" not in st.session_state:
        st.session_state.lit_table = _default_literature_table()
    edited = st.data_editor(
        st.session_state.lit_table,
        num_rows="dynamic",
        use_container_width=True,
        key="lit_editor",
    )
    st.session_state.lit_table = edited
    filled = _autofill_compared(edited, df_runs)
    st.dataframe(filled, use_container_width=True)
    st.download_button(
        "Download literature table (CSV)",
        dataframe_to_csv_bytes(filled),
        file_name="literature_comparison.csv",
        mime="text/csv",
    )

