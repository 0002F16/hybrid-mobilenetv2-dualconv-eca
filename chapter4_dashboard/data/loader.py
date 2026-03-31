from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from dashboard.loaders import load_disk_data


def load_disk_v2(repo_root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Load from repo-root `Trained Models v2/` using the existing (tested) loader.

    Note: `dashboard.loaders.load_disk_data("v2")` already implements:
      Trained Models v2/<dataset>/<variant>/seed_<seed>/metrics.json
      Trained Models v2/<dataset>/<variant>/seed_<seed>/logs/epochs.jsonl
    """
    # The existing loader resolves repo root internally; `repo_root` kept for API symmetry.
    try:
        df_runs, df_eff, df_curves, meta = load_disk_data("v2")
    except Exception as e:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            {"artifact_version": "v2", "error": str(e)},
        )
    return df_runs, df_eff, df_curves, meta

