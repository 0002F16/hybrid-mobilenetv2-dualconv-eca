"""Load metrics and curves from a single Trained Models v1/v2/v3 tree."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dashboard.constants import (
    DATASET_DISPLAY_TO_FOLDER,
    DATASET_FOLDER_TO_DISPLAY,
    DISPLAY_TO_FOLDER,
    VARIANT_FOLDER_TO_DISPLAY,
    VARIANT_ORDER,
    VERSION_TO_FOLDER,
    SEEDS,
    static_latency_ms,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _normalize_dataset(ds_raw: str) -> str:
    s = str(ds_raw).lower().replace("-", "_")
    mapping = {
        "cifar10": "CIFAR-10",
        "cifar100": "CIFAR-100",
        "tiny_imagenet": "Tiny-ImageNet",
    }
    return mapping.get(s, ds_raw)


def _parse_epochs_jsonl(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if "val_top1_pp" in df.columns:
        df["val_top1"] = df["val_top1_pp"]
    elif "val_acc" in df.columns:
        df["val_top1"] = df["val_acc"] * 100.0
    else:
        df["val_top1"] = np.nan
    return df


def load_disk_metrics_single(root: Path, folder_name: str) -> dict[tuple[str, str, int], dict[str, Any]]:
    """(dataset_display, variant_display, seed) -> metrics.json payload from one tree only."""
    out: dict[tuple[str, str, int], dict[str, Any]] = {}
    base = root / folder_name
    if not base.is_dir():
        return out
    for ds_folder, ds_disp in DATASET_FOLDER_TO_DISPLAY.items():
        ds_path = base / ds_folder
        if not ds_path.is_dir():
            continue
        for vf, vdisp in VARIANT_FOLDER_TO_DISPLAY.items():
            vpath = ds_path / vf
            if not vpath.is_dir():
                continue
            for seed in SEEDS:
                mj = vpath / f"seed_{seed}" / "metrics.json"
                if not mj.is_file():
                    continue
                key = (ds_disp, vdisp, seed)
                try:
                    out[key] = _read_json(mj)
                except (json.JSONDecodeError, OSError):
                    continue
    return out


def _parse_seed_folder_name(name: str) -> int | None:
    s = str(name)
    if not s.startswith("seed_"):
        return None
    try:
        return int(s.split("_", 1)[1])
    except ValueError:
        return None


def load_disk_records_v1(
    root: Path, folder_name: str
) -> dict[tuple[str, str, int], tuple[dict[str, Any], Path]]:
    """
    v1 layout (Colab export style) differs from v2/v3:

      Trained Models v1/<top>/content/outputs/<dataset>/<variant>/seed_x/metrics.json

    We discover metrics.json under any v1 <top> subtree, but only if the path contains
    'content/outputs' and matches the dataset/variant/seed conventions used by this repo.
    """
    out: dict[tuple[str, str, int], tuple[dict[str, Any], Path]] = {}
    base = root / folder_name
    if not base.is_dir():
        return out

    # Only accept these dataset folder names under content/outputs/.
    ds_folder_to_display = {
        "cifar10": "CIFAR-10",
        "cifar100": "CIFAR-100",
        "tiny_imagenet": "Tiny-ImageNet",
    }

    for mj in base.rglob("metrics.json"):
        try:
            parts = mj.parts
            if "content" not in parts or "outputs" not in parts:
                continue
            # ensure .../content/outputs/<dataset>/<variant>/seed_x/metrics.json
            i_out = parts.index("outputs")
            if i_out + 4 >= len(parts):
                continue
            ds_folder = parts[i_out + 1]
            var_folder = parts[i_out + 2]
            seed_folder = parts[i_out + 3]
            if parts[i_out - 1] != "content":
                continue
            if ds_folder not in ds_folder_to_display:
                continue
            if var_folder not in VARIANT_FOLDER_TO_DISPLAY:
                continue
            seed = _parse_seed_folder_name(seed_folder)
            if seed is None or seed not in SEEDS:
                continue

            # Prefer the dataset encoded in metrics.json when it matches known names.
            # This handles v1 exports where the outer folder name (e.g. "CIFAR10/") is not the true dataset.
            mj_payload = _read_json(mj)
            ds_disp = _normalize_dataset(mj_payload.get("dataset", ds_folder_to_display[ds_folder]))
            if ds_disp not in DATASET_DISPLAY_TO_FOLDER:
                ds_disp = ds_folder_to_display[ds_folder]
            vdisp = VARIANT_FOLDER_TO_DISPLAY[var_folder]
            key = (ds_disp, vdisp, seed)
            if key in out:
                # Keep the first one; v1 tree can include duplicates under different top folders.
                continue
            out[key] = (mj_payload, mj.parent)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
    return out


def _epochs_path(root: Path, folder_name: str, dataset: str, variant: str, seed: int) -> Path:
    ds_folder = DATASET_DISPLAY_TO_FOLDER[dataset]
    vf = DISPLAY_TO_FOLDER[variant]
    return root / folder_name / ds_folder / vf / f"seed_{seed}" / "logs" / "epochs.jsonl"


def _metrics_to_run_row(m: dict[str, Any]) -> dict[str, Any]:
    test = m.get("test") or {}
    top1 = float(test.get("top1_pp", (test.get("acc") or 0.0) * 100.0))
    top5 = float(test.get("top5_pp", 0.0))
    ds_raw = m.get("dataset") or ""
    dataset = _normalize_dataset(ds_raw)
    model = (m.get("model") or "baseline").lower().strip()
    vdisp = VARIANT_FOLDER_TO_DISPLAY.get(model, "Baseline")
    return {
        "seed": int(m.get("seed", 0)),
        "variant": vdisp,
        "dataset": dataset,
        "top1_acc": top1,
        "top5_acc": top5,
    }


def _efficiency_from_metrics(m: dict[str, Any]) -> dict[str, Any]:
    mp = m.get("model_profile") or {}
    params = float(mp.get("params", 0.0))
    flops = float(mp.get("flops", 0.0))
    size_mb = float(mp.get("size_mb", 0.0))
    model = (m.get("model") or "baseline").lower().strip()
    vdisp = VARIANT_FOLDER_TO_DISPLAY.get(model, "Baseline")
    return {
        "variant": vdisp,
        "params_M": params / 1e6,
        "flops_M": flops / 1e6,
        "size_mb": size_mb,
        "latency_ms": static_latency_ms(vdisp),
    }


def aggregate_efficiency(disk_metrics: dict[tuple[str, str, int], dict[str, Any]]) -> pd.DataFrame:
    """Mean params/flops/size per variant; latency from static table."""
    rows = [_efficiency_from_metrics(m) for m in disk_metrics.values()]
    if not rows:
        return pd.DataFrame(columns=["variant", "params_M", "flops_M", "size_mb", "latency_ms"])
    df = pd.DataFrame(rows)
    g = df.groupby("variant", as_index=False).mean(numeric_only=True)
    g["latency_ms"] = g["variant"].map({v: static_latency_ms(v) for v in VARIANT_ORDER})
    return g


def _long_curve_from_df(
    df: pd.DataFrame,
    dataset: str,
    variant: str,
    seed: int,
    curves_source: str,
) -> pd.DataFrame:
    if "epoch" not in df.columns:
        return pd.DataFrame()
    tl = df["train_loss"] if "train_loss" in df.columns else np.nan
    vl = df["val_loss"] if "val_loss" in df.columns else np.nan
    vt = df["val_top1"] if "val_top1" in df.columns else np.nan
    n = len(df)
    return pd.DataFrame(
        {
            "variant": variant,
            "dataset": dataset,
            "seed": seed,
            "epoch": df["epoch"].astype(int),
            "train_loss": tl,
            "val_loss": vl,
            "val_top1": vt,
            "curves_source": [curves_source] * n,
        }
    )


def load_disk_data(version: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if version not in VERSION_TO_FOLDER:
        raise ValueError(f"Unknown version {version!r}; expected one of {list(VERSION_TO_FOLDER)}")

    root = _repo_root()
    folder = VERSION_TO_FOLDER[version]
    seed_dirs_by_key: dict[tuple[str, str, int], Path] = {}
    if version == "v1":
        records = load_disk_records_v1(root, folder)
        disk_map = {k: v for k, (v, _p) in records.items()}
        seed_dirs_by_key = {k: p for k, (_v, p) in records.items()}
    else:
        disk_map = load_disk_metrics_single(root, folder)

    run_rows: list[dict[str, Any]] = []
    curve_parts: list[pd.DataFrame] = []

    for key, m in disk_map.items():
        ds, var, seed = key
        base_row = _metrics_to_run_row(m)
        if version == "v1":
            seed_dir = seed_dirs_by_key.get(key)
            ep = (seed_dir / "logs" / "epochs.jsonl") if seed_dir else Path("__missing__")
        else:
            ep = _epochs_path(root, folder, ds, var, seed)
        edf = _parse_epochs_jsonl(ep)
        curves_src = "missing"
        if edf is not None and len(edf) > 0 and "epoch" in edf.columns:
            long = _long_curve_from_df(edf, ds, var, seed, version)
            if len(long):
                curve_parts.append(long)
                curves_src = version
        run_rows.append(
            {
                **base_row,
                "metrics_source": version,
                "curves_source": curves_src,
            }
        )

    cols = ["seed", "variant", "dataset", "top1_acc", "top5_acc", "metrics_source", "curves_source"]
    df_runs = pd.DataFrame(run_rows, columns=cols) if run_rows else pd.DataFrame(columns=cols)

    if disk_map:
        df_eff = aggregate_efficiency(disk_map)
        df_eff = df_eff.copy()
        df_eff["metrics_json_from"] = version
    else:
        df_eff = pd.DataFrame(
            columns=["variant", "params_M", "flops_M", "size_mb", "latency_ms", "metrics_json_from"]
        )

    if curve_parts:
        df_curves = pd.concat(curve_parts, ignore_index=True)
    else:
        df_curves = pd.DataFrame(
            columns=["variant", "dataset", "seed", "epoch", "train_loss", "val_loss", "val_top1", "curves_source"]
        )

    n_missing_curves = (
        int((df_runs["curves_source"] == "missing").sum()) if len(df_runs) and "curves_source" in df_runs.columns else 0
    )
    meta = {
        "artifact_version": version,
        "artifact_folder": folder,
        "disk_run_count": len(disk_map),
        "metrics_by_version": df_runs["metrics_source"].value_counts().to_dict() if len(df_runs) else {},
        "curves_by_version": df_curves["curves_source"].value_counts().to_dict() if len(df_curves) else {},
        "runs_missing_curves": n_missing_curves,
        "tiny_imagenet_on_disk": bool((df_runs["dataset"] == "Tiny-ImageNet").any()) if len(df_runs) else False,
    }
    return df_runs, df_eff, df_curves, meta


def compute_data_hash(df_runs: pd.DataFrame, df_eff: pd.DataFrame, df_curves: pd.DataFrame) -> str:
    h = hashlib.sha256()
    for df in (df_runs, df_eff, df_curves):
        h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()


def parse_uploaded_csv(name: str, content: bytes) -> pd.DataFrame:
    import io

    try:
        return pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise ValueError(f"Failed to read {name}: {e}") from e
