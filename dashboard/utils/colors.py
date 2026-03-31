"""Color palettes and Plotly theme helpers."""

from __future__ import annotations

import colorsys
from typing import Literal

from dashboard.constants import VARIANT_ORDER

BASE_COLORS = {
    "Baseline": "#4C72B0",
    "DualConv-only": "#DD8452",
    "ECA-only": "#55A868",
    "Hybrid": "#C44E52",
}


def _hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def _rgb_to_hex(r: float, g: float, b: float) -> str:
    return "#{:02x}{:02x}{:02x}".format(
        int(max(0, min(1, r)) * 255),
        int(max(0, min(1, g)) * 255),
        int(max(0, min(1, b)) * 255),
    )


def _brighter(hex_color: str, factor: float = 0.3) -> str:
    r, g, b = _hex_to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = min(1.0, l + factor * (1.0 - l))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return _rgb_to_hex(r, g, b)


def _desaturate(hex_color: str, amount: float = 0.5) -> str:
    r, g, b = _hex_to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    s *= 1.0 - amount
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return _rgb_to_hex(r, g, b)


def get_theme_colors(
    theme: Literal["Publication", "Dark", "Pastel"],
) -> dict[str, str]:
    if theme == "Publication":
        return dict(BASE_COLORS)
    if theme == "Dark":
        return {k: _brighter(v, 0.35) for k, v in BASE_COLORS.items()}
    return {k: _desaturate(v, 0.5) for k, v in BASE_COLORS.items()}


def plotly_template(theme: Literal["Publication", "Dark", "Pastel"]) -> str:
    return "plotly_dark" if theme == "Dark" else "plotly_white"


def plotly_font_kwargs() -> dict:
    return {
        "font": {"family": "Arial", "size": 12},
        "title_font": {"family": "Arial", "size": 12},
    }


def axis_tick_font() -> dict:
    return {"tickfont": {"family": "Arial", "size": 10}}


def variant_color_map(theme: str) -> dict[str, str]:
    c = get_theme_colors(theme)  # type: ignore[arg-type]
    return {v: c[v] for v in VARIANT_ORDER if v in c}
