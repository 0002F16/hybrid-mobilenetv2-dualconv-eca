"""Figure PNG and table CSV export helpers."""

from __future__ import annotations

import io

import pandas as pd
import plotly.graph_objects as go


EXPORT_WIDTH = 1200
EXPORT_HEIGHT = 600


def figure_to_png_bytes(fig: go.Figure) -> bytes:
    fig = go.Figure(fig)
    fig.update_layout(width=EXPORT_WIDTH, height=EXPORT_HEIGHT)
    return fig.to_image(format="png", engine="kaleido", width=EXPORT_WIDTH, height=EXPORT_HEIGHT)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")
