"""Streamlit dashboard: Chapter 4 (Results and Discussion).

Separate app from the existing `app.py`.
"""

from __future__ import annotations

import streamlit as st

from chapter4_dashboard.app import render_app


def main() -> None:
    st.set_page_config(layout="wide", page_title="MobileNetV2 Hybrid — Results")
    render_app()


if __name__ == "__main__":
    main()

