from __future__ import annotations

import pandas as pd
import streamlit as st

from chapter4_dashboard.figures.curves import build_training_diagnostic_plot, convergence_summary_table
from chapter4_dashboard.figures.early_stop import build_early_stopping_distribution
from chapter4_dashboard.utils.constants import DATASET_ORDER, SEEDS
from chapter4_dashboard.utils.export import dataframe_to_csv_bytes, figure_to_png_bytes


def render_tab_diagnostics(df_curves: pd.DataFrame, df_runs: pd.DataFrame, cmap: dict[str, str]) -> None:
    st.subheader("Tab 6 — Training Diagnostics")
    st.caption(
        "Sanity-check training behavior. Not a primary results section but supports validity of reported endpoint metrics."
    )
    if df_curves.empty:
        st.info("No curve data available (`df_curves` is empty).")
        return

    ds = st.radio("Dataset", DATASET_ORDER, horizontal=True, key="diag_ds")
    metric = st.selectbox("Metric", ["Val Top-1", "Train Loss", "Val Loss"], key="diag_metric")
    view = st.radio("View mode", ["Single seed", "All seeds (mean ± band)"], horizontal=True, key="diag_view")
    seed = None
    if view == "Single seed":
        seed = st.selectbox("Seed", SEEDS, key="diag_seed")
    smoothing = st.slider("Smoothing", 1, 15, 3, key="diag_smooth")

    fig = build_training_diagnostic_plot(
        df_curves=df_curves,
        dataset=ds,
        metric=metric,
        view_mode=view.split(" (")[0],
        seed=int(seed) if seed is not None else None,
        smoothing=int(smoothing),
        cmap=cmap,
        theme=st.session_state.theme,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.download_button(
        "Download diagnostics plot (PNG)",
        figure_to_png_bytes(fig),
        file_name=f"training_diagnostics_{ds.replace('-', '_').lower()}.png",
        mime="image/png",
    )

    st.markdown("**Convergence Summary Table**")
    cs = convergence_summary_table(df_curves, ds)
    st.dataframe(cs, use_container_width=True)
    st.download_button(
        "Download convergence summary (CSV)",
        dataframe_to_csv_bytes(cs),
        file_name=f"convergence_summary_{ds.replace('-', '_').lower()}.csv",
        mime="text/csv",
    )

    st.markdown("**Early Stopping Distribution**")
    fig_es = build_early_stopping_distribution(df_curves, ds, cmap, st.session_state.theme)
    if fig_es.data:
        st.plotly_chart(fig_es, use_container_width=True)
        st.download_button(
            "Download early-stopping distribution (PNG)",
            figure_to_png_bytes(fig_es),
            file_name=f"early_stopping_{ds.replace('-', '_').lower()}.png",
            mime="image/png",
        )
    else:
        st.info("Early stopping distribution requires `is_best_epoch` in `df_curves`.")

