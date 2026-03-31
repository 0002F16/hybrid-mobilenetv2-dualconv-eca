from __future__ import annotations

import pandas as pd
import streamlit as st

from chapter4_dashboard.utils.constants import DATASET_ORDER, SEEDS, VARIANT_ORDER
from chapter4_dashboard.utils.styling import style_efficiency_delta, style_runs_raw


def render_tab_raw_data(
    df_runs: pd.DataFrame, df_eff: pd.DataFrame, df_curves: pd.DataFrame, cmap: dict[str, str]
) -> None:
    st.subheader("Tab 1 — Raw Data")
    st.caption("Inspect and verify data before analysis.")

    view = st.radio(
        "View",
        ["Per-Run Accuracy", "Efficiency", "Training Curves"],
        horizontal=True,
        key="raw_view",
    )

    if view == "Per-Run Accuracy":
        st.dataframe(style_runs_raw(df_runs), use_container_width=True, height=420)

        c1, c2 = st.columns(2)
        pv1 = df_runs.pivot_table(
            index="variant",
            columns="dataset",
            values="top1_acc",
            aggfunc=lambda x: f"{x.mean():.2f} ± {x.std():.2f}",
        )
        pv2 = df_runs.pivot_table(
            index="variant",
            columns="dataset",
            values="top5_acc",
            aggfunc=lambda x: f"{x.mean():.2f} ± {x.std():.2f}",
        )
        with c1:
            st.caption("Top-1 mean ± std")
            st.dataframe(pv1, use_container_width=True)
        with c2:
            st.caption("Top-5 mean ± std")
            st.dataframe(pv2, use_container_width=True)

    elif view == "Efficiency":
        if df_eff.empty:
            st.info("No efficiency data available.")
            return
        bl = df_eff[df_eff["variant"] == "Baseline"]
        if bl.empty:
            st.warning("No Baseline row in df_efficiency.")
            return
        base = bl.iloc[0]
        de = df_eff.copy()
        de["delta_params_pct"] = (de["params_M"] - base["params_M"]) / base["params_M"] * 100
        de["delta_flops_pct"] = (de["flops_M"] - base["flops_M"]) / base["flops_M"] * 100

        within = bool(((de["delta_params_pct"].abs() <= 10) & (de["delta_flops_pct"].abs() <= 10)).all())
        st.info(f'All variants within ±10% budget: **{"TRUE" if within else "FALSE"}**')
        st.dataframe(style_efficiency_delta(de), use_container_width=True, height=320)

    else:
        if df_curves.empty:
            st.info("No training curve data available.")
            return
        dsel = st.selectbox("Dataset", DATASET_ORDER, key="raw_curves_ds")
        vsel = st.multiselect("Variants", VARIANT_ORDER, default=VARIANT_ORDER, key="raw_curves_v")
        ssel = st.multiselect("Seeds", SEEDS, default=SEEDS, key="raw_curves_s")
        ep = st.slider("Epoch range", 1, 200, (1, 200), key="raw_curves_ep")
        sub = df_curves[
            (df_curves["dataset"] == dsel)
            & (df_curves["variant"].isin(vsel))
            & (df_curves["seed"].isin(ssel))
            & (df_curves["epoch"] >= ep[0])
            & (df_curves["epoch"] <= ep[1])
        ]
        st.dataframe(sub, use_container_width=True, height=420)

