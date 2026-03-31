from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from chapter4_dashboard.figures.paired import build_paired_delta_grid
from chapter4_dashboard.utils.export import dataframe_to_csv_bytes, figure_to_png_bytes
from chapter4_dashboard.utils.constants import SEEDS


def _mean_delta(df_runs: pd.DataFrame, dataset: str, variant: str) -> tuple[float, float]:
    base = df_runs[(df_runs["dataset"] == dataset) & (df_runs["variant"] == "Baseline")][["seed", "top1_acc"]].rename(
        columns={"top1_acc": "base"}
    )
    var = df_runs[(df_runs["dataset"] == dataset) & (df_runs["variant"] == variant)][["seed", "top1_acc"]].rename(
        columns={"top1_acc": "var"}
    )
    m = pd.merge(base, var, on="seed", how="inner")
    d = (m["var"].astype(float) - m["base"].astype(float)).to_numpy()
    return (float(np.mean(d)) if len(d) else float("nan"), float(np.std(d, ddof=1)) if len(d) > 1 else 0.0)


def _interaction_label(interaction_pp: float) -> str:
    if not np.isfinite(interaction_pp):
        return ""
    if interaction_pp > 0.1:
        return f"Synergistic ↑ (+{interaction_pp:.2f} pp)"
    if interaction_pp < -0.1:
        return f"Redundant ↓ ({interaction_pp:.2f} pp)"
    return "Additive (~0 pp)"


def render_tab_ablation(df_runs: pd.DataFrame, cmap: dict[str, str]) -> None:
    st.subheader("Tab 4 — Objective 3: Ablation Analysis")
    st.info(
        "“Specific Objective 3: Quantify the individual contributions of DualConv and ECA, and "
        "their combined interaction, relative to the Baseline model.”"
    )
    st.info(
        "CIFAR-10 is excluded from ablation interpretation. With high baseline accuracy (~92%) and low "
        "class complexity, per-seed variance (±0.4 pp) exceeds expected component deltas (~0.2–0.3 pp), "
        "making attribution unreliable. Analysis focuses on CIFAR-100 and Tiny-ImageNet."
    )

    ds_left, ds_right = "CIFAR-100", "Tiny-ImageNet"
    c1, c2 = st.columns(2)
    for col, ds in zip([c1, c2], [ds_left, ds_right]):
        with col:
            st.markdown(f"**4A — Δ Top-1 vs Baseline ({ds})**")
            rows = []
            for v in ["Baseline", "DualConv-only", "ECA-only", "Hybrid"]:
                if v == "Baseline":
                    mu, sd = 0.0, 0.0
                else:
                    mu, sd = _mean_delta(df_runs, ds, v)
                rows.append({"Variant": v, "Δ Top-1 (pp)": mu, "SD": sd})
            t = pd.DataFrame(rows)
            st.dataframe(
                t.style.format({"Δ Top-1 (pp)": "{:+.2f}", "SD": "{:.2f}"}),
                use_container_width=True,
                height=220,
            )
            st.download_button(
                f"Download ablation deltas ({ds}) (CSV)",
                dataframe_to_csv_bytes(t),
                file_name=f"ablation_deltas_{ds.replace('-', '_').lower()}.csv",
                mime="text/csv",
            )

            st.markdown("**4B — Component Attribution Metrics**")
            d_mu, _ = _mean_delta(df_runs, ds, "DualConv-only")
            e_mu, _ = _mean_delta(df_runs, ds, "ECA-only")
            h_mu, _ = _mean_delta(df_runs, ds, "Hybrid")
            interaction = h_mu - d_mu - e_mu
            m1, m2, m3 = st.columns(3)
            m1.metric("DualConv contribution", f"{d_mu:+.2f} pp")
            m2.metric("ECA contribution", f"{e_mu:+.2f} pp")
            m3.metric("Interaction term", _interaction_label(interaction))

    st.markdown("**4C — Per-Seed Paired Differences (most important diagnostic figure)**")
    fig = build_paired_delta_grid(
        df_runs,
        datasets=[ds_left, ds_right],
        variants=["DualConv-only", "ECA-only", "Hybrid"],
        theme=st.session_state.theme,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.download_button(
        "Download paired-differences grid (PNG)",
        figure_to_png_bytes(fig),
        file_name="paired_differences_grid.png",
        mime="image/png",
    )

    # Narrative table (auto-generated)
    st.markdown("**Narrative interpretation table**")
    tb = st.session_state.get("stats_results")
    rows = []
    for ds in ["CIFAR-10", ds_left, ds_right]:
        for v in ["DualConv-only", "ECA-only", "Hybrid"]:
            if ds == "CIFAR-10":
                rows.append(
                    {
                        "Variant": v,
                        "Dataset": ds,
                        "Seeds improved": "low confidence",
                        "Median Δ": "—",
                        "Corrected p": "—",
                        "Interpretation": "Low confidence",
                    }
                )
                continue
            base = df_runs[(df_runs["dataset"] == ds) & (df_runs["variant"] == "Baseline")][["seed", "top1_acc"]].rename(
                columns={"top1_acc": "base"}
            )
            var = df_runs[(df_runs["dataset"] == ds) & (df_runs["variant"] == v)][["seed", "top1_acc"]].rename(
                columns={"top1_acc": "var"}
            )
            m = pd.merge(base, var, on="seed", how="inner")
            deltas = (m["var"].astype(float) - m["base"].astype(float)).to_numpy()
            wins = int((deltas > 0).sum())
            med = float(np.median(deltas)) if len(deltas) else float("nan")
            pcorr = float("nan")
            if tb is not None and hasattr(tb, "__len__") and len(tb):
                sub = tb[(tb["Dataset"] == ds) & (tb["Variant"] == v) & (tb["Metric"] == "Top-1")]
                if len(sub):
                    pcorr = float(sub["Corrected p"].iloc[0])
            sig = np.isfinite(pcorr) and pcorr < 0.05
            if wins == 5 and sig:
                interp = "Consistent improvement"
            elif wins >= 4 and sig:
                interp = "Probable improvement"
            elif wins <= 1:
                interp = "Degradation"
            else:
                interp = "Inconclusive"
            rows.append(
                {
                    "Variant": v,
                    "Dataset": ds,
                    "Seeds improved": f"{wins}/5",
                    "Median Δ": f"{med:+.3f} pp",
                    "Corrected p": f"{pcorr:.4f}" if np.isfinite(pcorr) else "",
                    "Interpretation": interp,
                }
            )
    narr = pd.DataFrame(rows)
    st.dataframe(narr, use_container_width=True, height=360)
    st.download_button(
        "Download narrative table (CSV)",
        dataframe_to_csv_bytes(narr),
        file_name="ablation_narrative_table.csv",
        mime="text/csv",
    )

