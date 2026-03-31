from __future__ import annotations

from typing import Literal

from dashboard.utils.colors import BASE_COLORS, get_theme_colors, plotly_font_kwargs, plotly_template


def variant_color_map(theme: Literal["Publication", "Dark", "Pastel"]) -> dict[str, str]:
    # Keep exactly the requested base palette; theme variants are derived.
    return get_theme_colors(theme)

