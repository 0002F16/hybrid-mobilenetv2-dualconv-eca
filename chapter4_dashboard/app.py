from __future__ import annotations

import pandas as pd
import streamlit as st

from chapter4_dashboard.data.loader import load_disk_v2
from chapter4_dashboard.data.placeholder import generate_placeholder
from chapter4_dashboard.utils.captions import caption_table_main_results, caption_table_stats
from chapter4_dashboard.utils.colors import variant_color_map
from chapter4_dashboard.utils.export import (
    dataframe_to_csv_bytes,
    figure_to_png_bytes,
)
from chapter4_dashboard.utils.state import (
    ensure_session_state,
    get_filtered,
    recompute_if_needed,
)

from chapter4_dashboard.tabs.raw_data import render_tab_raw_data
from chapter4_dashboard.tabs.objective1_budget import render_tab_budget
from chapter4_dashboard.tabs.objective2_performance import render_tab_performance
from chapter4_dashboard.tabs.objective3_ablation import render_tab_ablation
from chapter4_dashboard.tabs.comparative_analysis import render_tab_comparative
from chapter4_dashboard.tabs.training_diagnostics import render_tab_diagnostics


def _init_data_if_needed() -> None:
    if "df_runs" in st.session_state and "df_efficiency" in st.session_state and "df_curves" in st.session_state:
        return

    df_runs, df_eff, df_curves, _meta = load_disk_v2()
    if len(df_runs) and len(df_eff):
        st.session_state.df_runs = df_runs
        st.session_state.df_efficiency = df_eff
        st.session_state.df_curves = df_curves
        st.session_state.data_source = "disk:v2"
    else:
        df_runs, df_eff, df_curves = generate_placeholder()
        st.session_state.df_runs = df_runs
        st.session_state.df_efficiency = df_eff
        st.session_state.df_curves = df_curves
        st.session_state.data_source = "placeholder"


def _sidebar_data_upload() -> None:
    st.sidebar.subheader("Data upload")
    with st.sidebar.expander("Schema (CSV columns)"):
        st.code(
            "df_runs: seed, variant, dataset, top1_acc, top5_acc\n"
            "df_efficiency: variant, params_M, flops_M, size_mb, latency_ms\n"
            "df_curves: variant, dataset, seed, epoch, train_loss, val_loss, val_top1, is_best_epoch"
        )

    up_runs = st.sidebar.file_uploader("Upload df_runs.csv", type=["csv"], key="up_runs")
    up_eff = st.sidebar.file_uploader("Upload df_efficiency.csv", type=["csv"], key="up_eff")
    up_curves = st.sidebar.file_uploader("Upload df_curves.csv", type=["csv"], key="up_curves")

    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        apply_clicked = st.button("Apply CSV uploads", use_container_width=True)
    with col_b:
        reset_clicked = st.button("Reset to placeholder data", use_container_width=True)

    if reset_clicked:
        df_runs, df_eff, df_curves = generate_placeholder()
        st.session_state.df_runs = df_runs
        st.session_state.df_efficiency = df_eff
        st.session_state.df_curves = df_curves
        st.session_state.data_source = "placeholder"
        st.session_state.stats_results = None
        st.rerun()

    if apply_clicked:
        if not (up_runs and up_eff and up_curves):
            st.sidebar.warning("Provide all three CSV files.")
            return
        try:
            df_runs = pd.read_csv(up_runs)
            df_eff = pd.read_csv(up_eff)
            df_curves = pd.read_csv(up_curves)
        except Exception as e:
            st.sidebar.error(f"Failed reading CSV(s): {e}")
            return

        st.session_state.df_runs = df_runs
        st.session_state.df_efficiency = df_eff
        st.session_state.df_curves = df_curves
        st.session_state.data_source = "csv"
        st.session_state.stats_results = None
        st.rerun()


def _sidebar_global_filters() -> None:
    st.sidebar.subheader("Global filters")
    st.sidebar.multiselect(
        "Datasets",
        st.session_state.all_datasets,
        default=st.session_state.selected_datasets,
        key="selected_datasets",
    )
    st.sidebar.multiselect(
        "Variants",
        st.session_state.all_variants,
        default=st.session_state.selected_variants,
        key="selected_variants",
    )

    st.sidebar.subheader("Appearance")
    st.sidebar.selectbox(
        "Color theme",
        ["Publication", "Dark", "Pastel"],
        index=["Publication", "Dark", "Pastel"].index(st.session_state.theme),
        key="theme",
    )
    st.sidebar.checkbox("Show 95% CI", value=bool(st.session_state.show_ci), key="show_ci")


def _sidebar_export_section(df_runs: pd.DataFrame, df_eff: pd.DataFrame, df_curves: pd.DataFrame) -> None:
    st.sidebar.subheader("Export (tables)")
    st.sidebar.download_button(
        "Download df_runs.csv",
        dataframe_to_csv_bytes(df_runs),
        "df_runs.csv",
        "text/csv",
        use_container_width=True,
    )
    st.sidebar.download_button(
        "Download df_efficiency.csv",
        dataframe_to_csv_bytes(df_eff),
        "df_efficiency.csv",
        "text/csv",
        use_container_width=True,
    )
    st.sidebar.download_button(
        "Download df_curves.csv",
        dataframe_to_csv_bytes(df_curves),
        "df_curves.csv",
        "text/csv",
        use_container_width=True,
    )


def render_app() -> None:
    ensure_session_state()
    _init_data_if_needed()

    df_runs = st.session_state.df_runs
    df_eff = st.session_state.df_efficiency
    df_curves = st.session_state.df_curves

    _sidebar_data_upload()
    _sidebar_global_filters()

    # Recompute heavy stats only on data change.
    recompute_if_needed()

    df_runs_f = get_filtered(df_runs)
    df_curves_f = get_filtered(df_curves)
    cmap = variant_color_map(st.session_state.theme)

    st.title("Chapter 4 — Results and Discussion")
    st.caption(f"**Data source:** {st.session_state.get('data_source', 'unknown')}")

    tabs = st.tabs(
        [
            "Raw Data",
            "Obj 1 — Budget Compliance",
            "Obj 2 — Performance Results",
            "Obj 3 — Ablation Analysis",
            "4.2 Comparative Analysis",
            "Training Diagnostics",
        ]
    )

    with tabs[0]:
        render_tab_raw_data(df_runs_f, df_eff, df_curves_f, cmap)
    with tabs[1]:
        render_tab_budget(df_eff, cmap)
    with tabs[2]:
        render_tab_performance(df_runs_f, df_eff, cmap)
    with tabs[3]:
        render_tab_ablation(df_runs_f, cmap)
    with tabs[4]:
        render_tab_comparative(df_runs_f, df_eff, cmap)
    with tabs[5]:
        render_tab_diagnostics(df_curves_f, df_runs_f, cmap)

