from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from chapter4_dashboard.stats.tests import compute_table_b
from chapter4_dashboard.utils.constants import DATASET_ORDER, VARIANT_ORDER


def compute_data_hash(df_runs: pd.DataFrame, df_eff: pd.DataFrame, df_curves: pd.DataFrame) -> str:
    h = hashlib.md5()
    for df in (df_runs, df_eff, df_curves):
        h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()


def ensure_session_state() -> None:
    st.session_state.setdefault("theme", "Publication")
    st.session_state.setdefault("show_ci", True)
    st.session_state.setdefault("all_datasets", list(DATASET_ORDER))
    st.session_state.setdefault("all_variants", list(VARIANT_ORDER))
    st.session_state.setdefault("selected_datasets", list(DATASET_ORDER))
    st.session_state.setdefault("selected_variants", list(VARIANT_ORDER))
    st.session_state.setdefault("stats_results", None)
    st.session_state.setdefault("data_hash", "")


def get_filtered(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    ds = st.session_state.get("selected_datasets", DATASET_ORDER)
    vs = st.session_state.get("selected_variants", VARIANT_ORDER)
    if "dataset" in df.columns:
        df = df[df["dataset"].isin(ds)]
    if "variant" in df.columns:
        df = df[df["variant"].isin(vs)]
    return df


@st.cache_data(show_spinner=False)
def _cached_table_b(df_runs: pd.DataFrame) -> pd.DataFrame:
    return compute_table_b(df_runs)


def recompute_if_needed() -> None:
    df_runs = st.session_state.df_runs
    df_eff = st.session_state.df_efficiency
    df_curves = st.session_state.df_curves
    new_hash = compute_data_hash(df_runs, df_eff, df_curves)
    if new_hash != st.session_state.get("data_hash", ""):
        st.session_state.data_hash = new_hash
        st.session_state.stats_results = _cached_table_b(df_runs)

