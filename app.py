"""Streamlit dashboard: MobileNetV2 architectural variants across datasets."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.constants import DATASET_ORDER, SEEDS, VARIANT_ORDER
from dashboard.figures.ablation import build_ablation_figure, interaction_metrics
from dashboard.figures.bars import build_cross_dataset_bars
from dashboard.figures.curves import build_training_curves, convergence_summary
from dashboard.figures.paired import build_paired_matrix
from dashboard.figures.radar import build_radar_figure
from dashboard.figures.scatter import build_accuracy_efficiency_scatter
from dashboard.loaders import compute_data_hash, load_disk_data, parse_uploaded_csv
from dashboard.stats.tests import build_table_b, count_seed_wins, get_corrected_p_for_pair
from dashboard.styling import (
    style_efficiency_delta,
    style_main_results_table,
    style_runs_raw,
)
from dashboard.utils.colors import get_theme_colors, variant_color_map
from dashboard.utils.export import EXPORT_HEIGHT, EXPORT_WIDTH, dataframe_to_csv_bytes, figure_to_png_bytes

st.set_page_config(layout="wide", page_title="ML Experiment Dashboard")

REQUIRED_RUNS = {"seed", "variant", "dataset", "top1_acc", "top5_acc"}
REQUIRED_EFF = {"variant", "params_M", "flops_M", "size_mb", "latency_ms"}
REQUIRED_CURVES = {"variant", "dataset", "seed", "epoch", "train_loss", "val_loss", "val_top1"}


@st.cache_data(show_spinner=False)
def cached_table_b(df_runs: pd.DataFrame) -> pd.DataFrame:
    return build_table_b(df_runs)


def sync_disk_session(version: str) -> None:
    df_r, df_e, df_c, meta = load_disk_data(version)
    st.session_state.df_runs = df_r
    st.session_state.df_efficiency = df_e
    st.session_state.df_curves = df_c
    st.session_state.meta = meta
    st.session_state.data_hash = compute_data_hash(df_r, df_e, df_c)
    st.session_state.upload_mode = False
    st.session_state.loaded_disk_version = version


def apply_uploads(runs_f, eff_f, cur_f):
    runs_f = runs_f.copy()
    eff_f = eff_f.copy()
    cur_f = cur_f.copy()
    if "metrics_source" not in runs_f.columns:
        runs_f["metrics_source"] = "CSV upload"
    if "curves_source" not in runs_f.columns:
        runs_f["curves_source"] = "CSV upload"
    if "curves_source" not in cur_f.columns:
        cur_f["curves_source"] = "CSV upload"
    if "metrics_json_from" not in eff_f.columns:
        eff_f["metrics_json_from"] = "CSV upload"
    st.session_state.df_runs = runs_f
    st.session_state.df_efficiency = eff_f
    st.session_state.df_curves = cur_f
    st.session_state.data_hash = compute_data_hash(runs_f, eff_f, cur_f)
    st.session_state.upload_mode = True
    disk_m = runs_f["metrics_source"].isin(["v1", "v2", "v3"])
    disk_c = runs_f["curves_source"].isin(["v1", "v2", "v3"])
    st.session_state.meta = {
        "tiny_imagenet_source": "csv",
        "disk_run_count": 0,
        "metrics_by_version": runs_f["metrics_source"].value_counts().to_dict(),
        "curves_by_version": cur_f["curves_source"].value_counts().to_dict(),
        "runs_split_metrics_vs_curves_version": int(
            (disk_m & disk_c & (runs_f["metrics_source"] != runs_f["curves_source"])).sum()
        ),
        "source": "csv",
    }


def filter_sidebar(df: pd.DataFrame, datasets: list[str], variants: list[str]) -> pd.DataFrame:
    return df[df["dataset"].isin(datasets) & df["variant"].isin(variants)]


def build_table_a(
    df_runs: pd.DataFrame,
    df_eff: pd.DataFrame,
    df_runs_for_baseline: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if df_runs.empty or df_eff.empty or "Baseline" not in df_eff["variant"].values:
        return pd.DataFrame()
    baseline_src = df_runs_for_baseline if df_runs_for_baseline is not None else df_runs
    base_rows = baseline_src[baseline_src["variant"] == "Baseline"]
    if base_rows.empty:
        return pd.DataFrame()
    baseline_means = base_rows.groupby("dataset", as_index=False).agg(
        base_top1=("top1_acc", "mean"),
        base_top5=("top5_acc", "mean"),
    )
    baseline_map = baseline_means.set_index("dataset")
    agg = (
        df_runs.groupby(["variant", "dataset"], as_index=False)
        .agg(
            top1_mean=("top1_acc", "mean"),
            top1_std=("top1_acc", "std"),
            top5_mean=("top5_acc", "mean"),
            top5_std=("top5_acc", "std"),
        )
    )
    eff = df_eff.set_index("variant")
    base_p = float(eff.loc["Baseline", "params_M"])
    base_f = float(eff.loc["Baseline", "flops_M"])
    rows = []
    for _, r in agg.iterrows():
        v = r["variant"]
        ds = r["dataset"]
        er = eff.loc[v]
        dpp = (float(er["params_M"]) - base_p) / base_p * 100.0
        dff = (float(er["flops_M"]) - base_f) / base_f * 100.0
        if ds in baseline_map.index:
            b1 = float(baseline_map.loc[ds, "base_top1"])
            b5 = float(baseline_map.loc[ds, "base_top5"])
            d_top1_pp = float(r["top1_mean"]) - b1
            d_top5_pp = float(r["top5_mean"]) - b5
        else:
            d_top1_pp = float("nan")
            d_top5_pp = float("nan")
        rows.append(
            {
                "Variant": v,
                "Dataset": ds,
                "top1_mean": r["top1_mean"],
                "top1_std": r["top1_std"],
                "top5_mean": r["top5_mean"],
                "top5_std": r["top5_std"],
                "ΔTop-1 pp": d_top1_pp,
                "ΔTop-5 pp": d_top5_pp,
                "Params(M)": float(er["params_M"]),
                "ΔParams%": dpp,
                "FLOPs(M)": float(er["flops_M"]),
                "ΔFLOPs%": dff,
                "Size(MB)": float(er["size_mb"]),
                "Latency(ms)": float(er["latency_ms"]),
            }
        )
    return pd.DataFrame(rows)


def main():
    upload_mode = st.session_state.get("upload_mode", False)

    st.sidebar.subheader("Model artifacts")
    if upload_mode:
        st.sidebar.caption(
            "Using **uploaded CSV**. Pick **Trained Models v1/**, **v2/**, or **v3/** below and click **Reset to disk** "
            "to load from disk again."
        )
    version_choice = st.sidebar.radio(
        "Load from",
        ["v1", "v2", "v3"],
        index={"v1": 0, "v2": 1, "v3": 2}.get(st.session_state.get("loaded_disk_version", "v3"), 2),
        format_func=lambda x: f"Trained Models {x}/",
        horizontal=True,
        disabled=upload_mode,
        key="disk_version_radio",
    )

    if not upload_mode:
        if (
            "df_runs" not in st.session_state
            or st.session_state.get("loaded_disk_version") != version_choice
        ):
            sync_disk_session(version_choice)
    elif "df_runs" not in st.session_state:
        st.session_state.upload_mode = False
        sync_disk_session(version_choice)

    df_runs = st.session_state.df_runs
    df_eff = st.session_state.df_efficiency
    df_curves = st.session_state.df_curves
    theme = st.sidebar.selectbox("Color theme", ["Publication", "Dark", "Pastel"], index=0)
    cmap = variant_color_map(theme)

    datasets_sel = st.sidebar.multiselect(
        "Datasets", DATASET_ORDER, default=DATASET_ORDER
    )
    variants_sel = st.sidebar.multiselect(
        "Variants", VARIANT_ORDER, default=VARIANT_ORDER
    )
    show_ci = st.sidebar.checkbox("Show 95% bootstrap CI on bar charts", value=False)

    meta = st.session_state.get("meta", {})
    st.sidebar.markdown("---")
    st.sidebar.subheader("Data provenance")
    if upload_mode:
        st.sidebar.info(
            "**Source:** uploaded CSV. `metrics_source` / `curves_source` "
            "default to **CSV upload** if those columns are missing."
        )
    else:
        folder = meta.get("artifact_folder", "")
        mb = meta.get("metrics_by_version", {})
        cb = meta.get("curves_by_version", {})
        rc = meta.get("tiny_imagenet_on_disk", False)
        m_s = ", ".join(f"**{k}**: {v}" for k, v in sorted(mb.items()))
        c_s = ", ".join(f"**{k}**: {v}" for k, v in sorted(cb.items()))
        nmiss = meta.get("runs_missing_curves", 0)
        miss_line = f"\n\n**Runs without usable `epochs.jsonl`:** {nmiss}" if nmiss else ""
        st.sidebar.info(
            f"**Folder:** `{folder}/`\n\n"
            f"**`metrics.json` (per-run):** {m_s}\n\n"
            f"**`epochs.jsonl` (curve rows):** {c_s}"
            + miss_line
            + (
                f"\n\n**Tiny-ImageNet on disk:** yes"
                if rc
                else "\n\n**Tiny-ImageNet on disk:** no (upload CSV or add runs under this folder)"
            )
        )

    st.sidebar.markdown("---")
    st.sidebar.subheader("CSV upload")
    with st.sidebar.expander("Schema (required columns)"):
        st.code(
            "df_runs: seed, variant, dataset, top1_acc, top5_acc\n"
            "  optional: metrics_source, curves_source (else ‘CSV upload’)\n"
            "df_efficiency: variant, params_M, flops_M, size_mb, latency_ms\n"
            "  optional: metrics_json_from\n"
            "df_curves: variant, dataset, seed, epoch, train_loss, val_loss, val_top1\n"
            "  optional: curves_source per row (else ‘CSV upload’)"
        )
    up_runs = st.sidebar.file_uploader("Upload df_runs.csv", type=["csv"])
    up_eff = st.sidebar.file_uploader("Upload df_efficiency.csv", type=["csv"])
    up_cur = st.sidebar.file_uploader("Upload df_curves.csv", type=["csv"])
    if st.sidebar.button("Apply CSV uploads"):
        if up_runs and up_eff and up_cur:
            try:
                dr = parse_uploaded_csv("runs", up_runs.getvalue())
                de = parse_uploaded_csv("eff", up_eff.getvalue())
                dc = parse_uploaded_csv("curves", up_cur.getvalue())
                if not REQUIRED_RUNS.issubset(set(dr.columns)):
                    st.sidebar.error(f"df_runs missing columns: {REQUIRED_RUNS - set(dr.columns)}")
                elif not REQUIRED_EFF.issubset(set(de.columns)):
                    st.sidebar.error("df_efficiency schema mismatch")
                elif not REQUIRED_CURVES.issubset(set(dc.columns)):
                    st.sidebar.error("df_curves schema mismatch")
                else:
                    apply_uploads(dr, de, dc)
                    st.sidebar.success("Replaced session data from CSV.")
                    st.rerun()
            except ValueError as e:
                st.sidebar.error(str(e))
        else:
            st.sidebar.warning("Provide all three CSV files.")

    if st.sidebar.button("Reset to disk (selected version)"):
        sync_disk_session(st.session_state.get("disk_version_radio", "v3"))
        st.rerun()

    df_f = filter_sidebar(df_runs, datasets_sel, variants_sel)

    if upload_mode:
        st.caption(
            "**Provenance:** Data from **uploaded CSV**. Version labels are user-supplied or defaulted to “CSV upload”."
        )
    else:
        av = meta.get("artifact_version", version_choice)
        st.caption(
            f"**Provenance:** Metrics and curves load only from **`{meta.get('artifact_folder', '')}/`**. "
            f"Per-run **`metrics_source`** and **`curves_source`** (and **`metrics_json_from`** on efficiency) "
            f"are **`{av}`** for disk metrics; **`missing`** means no usable `epochs.jsonl` for that run."
        )

    tabs = st.tabs(
        [
            "Raw Data",
            "Main Results Table",
            "Accuracy–Efficiency",
            "Cross-Dataset Bars",
            "Paired Difference",
            "Ablation Delta",
            "Training Curves",
        ]
    )

    # Tab 1
    with tabs[0]:
        sub = st.radio("View", ["Per-Run Results", "Efficiency", "Training Curves"], horizontal=True)
        if sub == "Per-Run Results":
            st.caption(
                "**`metrics_source`**: artifact set for `metrics.json` (test Top-1/5). "
                "**`curves_source`**: same for `epochs.jsonl`, or **`missing`** if no valid log."
            )
            st.dataframe(style_runs_raw(df_f), use_container_width=True, height=420)
            c1, c2 = st.columns(2)
            pv1 = df_f.pivot_table(
                index="variant",
                columns="dataset",
                values="top1_acc",
                aggfunc=lambda x: f"{x.mean():.2f} ± {x.std():.3f}",
            )
            pv2 = df_f.pivot_table(
                index="variant",
                columns="dataset",
                values="top5_acc",
                aggfunc=lambda x: f"{x.mean():.2f} ± {x.std():.3f}",
            )
            with c1:
                st.caption("Top-1 mean ± std")
                st.dataframe(pv1, use_container_width=True)
            with c2:
                st.caption("Top-5 mean ± std")
                st.dataframe(pv2, use_container_width=True)
        elif sub == "Efficiency":
            st.caption(
                "**`metrics_json_from`**: which artifact tree averaged for model profiles (**`v2`** or **`v3`**)."
            )
            bl = df_eff[df_eff["variant"] == "Baseline"]
            if bl.empty:
                st.warning("No Baseline row in efficiency table.")
            else:
                base = bl.iloc[0]
                de = df_eff.copy()
                de["delta_params_pct"] = (de["params_M"] - base["params_M"]) / base["params_M"] * 100
                de["delta_flops_pct"] = (de["flops_M"] - base["flops_M"]) / base["flops_M"] * 100
                st.dataframe(style_efficiency_delta(de), use_container_width=True)
        else:
            dsel = st.selectbox("Dataset", datasets_sel or DATASET_ORDER, key="tc_ds")
            vsel = st.selectbox("Variant", variants_sel or VARIANT_ORDER, key="tc_v")
            ssel = st.selectbox("Seed", SEEDS, key="tc_s")
            ep = st.slider("Epoch range", 1, 200, (1, 200))
            subc = df_curves[
                (df_curves["dataset"] == dsel)
                & (df_curves["variant"] == vsel)
                & (df_curves["seed"] == ssel)
                & (df_curves["epoch"] >= ep[0])
                & (df_curves["epoch"] <= ep[1])
            ]
            st.caption("**`curves_source`**: **`v2`** / **`v3`** from disk for that run’s `epochs.jsonl`.")
            st.dataframe(subc, use_container_width=True, height=400)

    # Tab 2
    with tabs[1]:
        st.caption(
            "Table A aggregates **Top-1/5** over `df_runs` (check **`metrics_source`** per run). "
            "**Params / FLOPs** come from **`metrics_json_from`** profiles on the efficiency table."
        )
        tbl = build_table_a(df_f, df_eff, df_runs_for_baseline=df_runs)
        if tbl.empty:
            st.info(
                "Table A needs runs, **Baseline** runs per dataset for accuracy deltas, and a **Baseline** "
                "efficiency row. Adjust sidebar filters or load disk data / upload CSV."
            )
        else:
            st.markdown("**Table A — Main results**")
            st.caption(
                "**ΔTop-1 / ΔTop-5** are mean accuracy minus **Baseline** mean on that dataset (percentage points). "
                "**ΔParams% / ΔFLOPs%** are vs Baseline on the efficiency table."
            )
            table_a_cols = [
                "Variant",
                "Top-1 mean±std",
                "ΔTop-1 pp",
                "Top-5 mean±std",
                "ΔTop-5 pp",
                "Params(M)",
                "ΔParams%",
                "FLOPs(M)",
                "ΔFLOPs%",
                "Size(MB)",
                "Latency(ms)",
                "top1_mean",
                "top1_std",
                "top5_mean",
                "top5_std",
            ]
            fmt_a = {
                "ΔTop-1 pp": "{:+.2f}",
                "ΔTop-5 pp": "{:+.2f}",
                "Params(M)": "{:.2f}",
                "ΔParams%": "{:.2f}",
                "FLOPs(M)": "{:.2f}",
                "ΔFLOPs%": "{:.2f}",
                "Size(MB)": "{:.2f}",
                "Latency(ms)": "{:.2f}",
            }
            for ds in [d for d in DATASET_ORDER if d in tbl["Dataset"].values]:
                sub = tbl[tbl["Dataset"] == ds].copy()
                disp = sub.assign(
                    **{
                        "Top-1 mean±std": sub.apply(
                            lambda r: f"{r['top1_mean']:.2f} ± {r['top1_std']:.2f}", axis=1
                        ),
                        "Top-5 mean±std": sub.apply(
                            lambda r: f"{r['top5_mean']:.2f} ± {r['top5_std']:.2f}", axis=1
                        ),
                    }
                )
                disp = disp[table_a_cols].sort_values("Variant")
                st.markdown(f"**{ds}**")
                styled_a = style_main_results_table(disp).hide(
                    subset=["top1_mean", "top1_std", "top5_mean", "top5_std"],
                    axis="columns",
                )
                styled_a = styled_a.format(fmt_a)
                st.dataframe(styled_a, use_container_width=True, height=min(120 + 42 * len(disp), 480))

        st.info(
            "n=5 seeds limits Wilcoxon power. Effect sizes and CIs are primary evidence."
        )
        tb = cached_table_b(df_runs)
        st.markdown("**Table B — Statistical tests (Wilcoxon vs Baseline, Holm–Bonferroni)**")
        b_cols = [
            "variant",
            "dataset",
            "metric",
            "median_delta",
            "ci_lower",
            "ci_upper",
            "W_statistic",
            "raw_p",
            "corrected_p",
            "significant",
        ]
        disp_b_rename = [
            "Variant",
            "Dataset",
            "Metric",
            "Median Δ",
            "95% CI lower",
            "95% CI upper",
            "W statistic",
            "Raw p",
            "Corrected p",
            "Significant",
        ]
        tb_display = tb[
            tb["dataset"].isin(datasets_sel) & tb["variant"].isin(variants_sel)
        ]
        for ds in [d for d in DATASET_ORDER if d in tb_display["dataset"].values]:
            disp_b = tb_display[tb_display["dataset"] == ds][b_cols].copy()
            disp_b.columns = disp_b_rename
            disp_b = disp_b.drop(columns=["Dataset"])
            st.markdown(f"**{ds}**")
            st.dataframe(disp_b, use_container_width=True, height=min(120 + 36 * len(disp_b), 420))

    # Tab 3
    with tabs[2]:
        st.caption(
            "FLOPs / size from **`df_efficiency`** (`metrics_json_from`); scatter points use **mean Top-1** from **`df_runs`** (`metrics_source`)."
        )
        fig_s = build_accuracy_efficiency_scatter(
            df_runs, df_eff, [d for d in DATASET_ORDER if d in datasets_sel], theme
        )
        fig_s.update_layout(**{"width": EXPORT_WIDTH, "height": EXPORT_HEIGHT})
        st.plotly_chart(fig_s, use_container_width=True)
        st.download_button(
            "Download scatter (PNG)",
            figure_to_png_bytes(fig_s),
            file_name="accuracy_efficiency_scatter.png",
            mime="image/png",
        )
        norm_radar = st.radio(
            "Radar normalization",
            ["Show normalized (÷ Baseline)", "Show absolute values"],
            horizontal=True,
        )
        fig_r = build_radar_figure(
            df_runs, df_eff, theme, normalized=("normalized" in norm_radar)
        )
        fig_r.update_layout(**{"width": EXPORT_WIDTH, "height": EXPORT_HEIGHT})
        st.plotly_chart(fig_r, use_container_width=True)
        st.download_button(
            "Download radar (PNG)",
            figure_to_png_bytes(fig_r),
            file_name="efficiency_radar.png",
            mime="image/png",
        )
        if "absolute" in norm_radar.lower():
            st.caption(
                "Absolute radar uses mixed units on one polar chart; interpret with the efficiency table."
            )

    # Tab 4
    with tabs[3]:
        st.caption("Bar heights follow **`df_runs`**; each row’s **`metrics_source`** labels which `metrics.json` produced that seed’s Top-1/5.")
        mode = st.radio("View", ["Absolute accuracy", "Delta vs Baseline"], horizontal=True)
        delta_mode = "Delta" in mode
        fig_b1 = build_cross_dataset_bars(
            df_runs,
            [d for d in DATASET_ORDER if d in datasets_sel],
            [v for v in VARIANT_ORDER if v in variants_sel],
            "top1",
            cmap,
            theme,
            show_ci,
            delta_mode,
        )
        fig_b1.update_layout(**{"width": EXPORT_WIDTH, "height": EXPORT_HEIGHT})
        st.plotly_chart(fig_b1, use_container_width=True)
        fig_b2 = build_cross_dataset_bars(
            df_runs,
            [d for d in DATASET_ORDER if d in datasets_sel],
            [v for v in VARIANT_ORDER if v in variants_sel],
            "top5",
            cmap,
            theme,
            show_ci,
            delta_mode,
        )
        fig_b2.update_layout(**{"width": EXPORT_WIDTH, "height": EXPORT_HEIGHT})
        st.plotly_chart(fig_b2, use_container_width=True)
        eb1, eb2 = st.columns(2)
        eb1.download_button(
            "Download Top-1 bars (PNG)",
            figure_to_png_bytes(fig_b1),
            file_name="cross_dataset_top1.png",
            mime="image/png",
        )
        eb2.download_button(
            "Download Top-5 bars (PNG)",
            figure_to_png_bytes(fig_b2),
            file_name="cross_dataset_top5.png",
            mime="image/png",
        )

    # Tab 5
    with tabs[4]:
        st.caption(
            "Paired differences use **`metrics_source`** accuracies (same seed, variant minus Baseline for that seed)."
        )
        fig_p, _, _ = build_paired_matrix(df_runs, [d for d in DATASET_ORDER if d in datasets_sel], theme)
        fig_p.update_layout(**{"width": EXPORT_WIDTH, "height": EXPORT_HEIGHT})
        st.plotly_chart(fig_p, use_container_width=True)
        st.download_button(
            "Download paired difference (PNG)",
            figure_to_png_bytes(fig_p),
            file_name="paired_difference.png",
            mime="image/png",
        )
        tb_stats = cached_table_b(df_runs)
        summaries = []
        for d in [x for x in DATASET_ORDER if x in datasets_sel]:
            for v in ["DualConv-only", "ECA-only", "Hybrid"]:
                wins, tot = count_seed_wins(df_runs, d, v, "top1")
                med = tb_stats.loc[
                    (tb_stats["variant"] == v)
                    & (tb_stats["dataset"] == d)
                    & (tb_stats["metric"] == "top1"),
                    "median_delta",
                ]
                med_v = float(med.iloc[0]) if len(med) else float("nan")
                pcorr = get_corrected_p_for_pair(tb_stats, v, d, "top1")
                summaries.append(
                    f"On **{d}**, **{v}** beats Baseline in **{wins}/{tot}** seeds "
                    f"(median Δ = {med_v:+.3f} pp, corrected p = {pcorr:.4f})."
                )
        st.markdown("\n\n".join(summaries))

    # Tab 6
    with tabs[5]:
        st.caption(
            "CIFAR-10 differences are often noise-level; we emphasize CIFAR-100 and Tiny-ImageNet. "
            "Bars use **`metrics_source`** test accuracies. **Tiny-ImageNet** appears only if present in `df_runs`."
        )
        c1, c2 = st.columns(2)
        with c1:
            f1 = build_ablation_figure(df_runs, "CIFAR-100", cmap, theme)
            f1.update_layout(**{"width": EXPORT_WIDTH // 2, "height": EXPORT_HEIGHT // 2})
            st.plotly_chart(f1, use_container_width=True)
        with c2:
            f2 = build_ablation_figure(df_runs, "Tiny-ImageNet", cmap, theme)
            f2.update_layout(**{"width": EXPORT_WIDTH // 2, "height": EXPORT_HEIGHT // 2})
            st.plotly_chart(f2, use_container_width=True)
        ab1, ab2 = st.columns(2)
        ab1.download_button(
            "Download ablation CIFAR-100 (PNG)",
            figure_to_png_bytes(f1),
            file_name="ablation_cifar100.png",
            mime="image/png",
        )
        ab2.download_button(
            "Download ablation Tiny-ImageNet (PNG)",
            figure_to_png_bytes(f2),
            file_name="ablation_tiny_imagenet.png",
            mime="image/png",
        )
        d1, e1, _z1, l1 = interaction_metrics(df_runs, "CIFAR-100")
        d2, e2, _z2, l2 = interaction_metrics(df_runs, "Tiny-ImageNet")
        st.markdown("**CIFAR-100**")
        a1, a2, a3 = st.columns(3)
        a1.metric("DualConv contribution", f"{d1:.2f} pp")
        a2.metric("ECA contribution", f"{e1:.2f} pp")
        a3.metric("Interaction", l1)
        st.markdown("**Tiny-ImageNet**")
        b1, b2, b3 = st.columns(3)
        b1.metric("DualConv contribution", f"{d2:.2f} pp")
        b2.metric("ECA contribution", f"{e2:.2f} pp")
        b3.metric("Interaction", l2)

    # Tab 7
    with tabs[6]:
        st.caption(
            "Curves use **`df_curves`** only from **`epochs.jsonl`** on disk; **`curves_source`** is **`v2`** or **`v3`**."
        )
        ds7 = st.radio("Dataset", DATASET_ORDER, horizontal=True, key="t7ds")
        all_seeds = st.checkbox("Show all seeds", value=False)
        seed7 = st.selectbox("Seed", SEEDS, key="t7seed") if not all_seeds else None
        metric7 = st.selectbox("Metric", ["Val Top-1", "Train Loss", "Val Loss"])
        win = st.slider("Smoothing window", 1, 20, 3)
        fig_c = build_training_curves(
            df_curves,
            ds7,
            seed7,
            all_seeds,
            metric7,
            win,
            cmap,
            theme,
        )
        fig_c.update_layout(**{"width": EXPORT_WIDTH, "height": EXPORT_HEIGHT})
        st.plotly_chart(fig_c, use_container_width=True)
        st.download_button(
            "Download training curves (PNG)",
            figure_to_png_bytes(fig_c),
            file_name="training_curves.png",
            mime="image/png",
        )
        cs = convergence_summary(df_curves, ds7)
        st.dataframe(cs, use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Export tables (CSV)")
    st.sidebar.download_button(
        "df_runs.csv",
        dataframe_to_csv_bytes(df_runs),
        "df_runs.csv",
        "text/csv",
    )
    st.sidebar.download_button(
        "df_efficiency.csv",
        dataframe_to_csv_bytes(df_eff),
        "df_efficiency.csv",
        "text/csv",
    )
    st.sidebar.download_button(
        "df_curves.csv",
        dataframe_to_csv_bytes(df_curves),
        "df_curves.csv",
        "text/csv",
    )
    tcsv = build_table_a(df_f, df_eff, df_runs_for_baseline=df_runs)
    st.sidebar.download_button(
        "table_a_main_results.csv",
        dataframe_to_csv_bytes(tcsv),
        "table_a_main_results.csv",
        "text/csv",
    )
    tb_export = cached_table_b(df_runs)
    st.sidebar.download_button(
        "table_b_statistics.csv",
        dataframe_to_csv_bytes(tb_export),
        "table_b_statistics.csv",
        "text/csv",
    )


if __name__ == "__main__":
    main()
