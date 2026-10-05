"""components/theme.py — dark-glassmorphism theme shared by every page."""
from __future__ import annotations

import streamlit as st

BG = "#0b0f19"
PANEL = "rgba(255,255,255,0.05)"
BORDER = "rgba(255,255,255,0.10)"
TEXT = "#e6e8ef"
MUTED = "#9aa1b2"
ACCENT = "#7c9cff"
SUCCESS = "#34d399"
WARNING = "#fbbf24"
DANGER = "#f87171"


def inject_css():
    st.markdown(
        f"""
        <style>
        .stApp {{ background: radial-gradient(1200px 800px at 20% -10%, #17203a 0%, {BG} 55%); color: {TEXT}; }}
        section[data-testid="stSidebar"] {{ background: #0d1220; border-right: 1px solid {BORDER}; }}
        .aeg-card {{
            background: {PANEL}; backdrop-filter: blur(10px); border: 1px solid {BORDER};
            border-radius: 14px; padding: 18px 20px; margin-bottom: 14px;
        }}
        .aeg-kpi-value {{ font-size: 28px; font-weight: 700; color: {TEXT}; }}
        /* UI-01: every KPI card in a row has the same height; labels stay on one
           line (full text on hover) and the help line sits at the bottom. */
        .aeg-kpi {{ min-height: 132px; display: flex; flex-direction: column; }}
        .aeg-kpi-help {{ color: {MUTED}; font-size: 12px; margin-top: auto; padding-top: 6px; line-height: 1.35; }}
        .aeg-kpi-label {{ white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 12px; color: {MUTED}; text-transform: uppercase; letter-spacing: .06em; }}
        .aeg-badge {{ display:inline-block; padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight:600; }}
        .aeg-badge-low {{ background: {SUCCESS}22; color:{SUCCESS}; border:1px solid {SUCCESS}55; }}
        .aeg-badge-medium {{ background: {WARNING}22; color:{WARNING}; border:1px solid {WARNING}55; }}
        .aeg-badge-high {{ background: {DANGER}22; color:{DANGER}; border:1px solid {DANGER}55; }}
        .aeg-note {{ background: {ACCENT}14; border: 1px solid {ACCENT}55; border-radius: 10px; padding: 12px 16px; }}
        h1, h2, h3 {{ color: {TEXT}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str):
    st.markdown(f"## {title}")
    st.markdown(f"<span style='color:{MUTED}'>{subtitle}</span>", unsafe_allow_html=True)
    st.write("")


def risk_badge(grade: str) -> str:
    cls = {"Low": "aeg-badge-low", "Medium": "aeg-badge-medium", "High": "aeg-badge-high"}.get(grade, "aeg-badge-medium")
    return f'<span class="aeg-badge {cls}">{grade}</span>'


def kpi_card(label: str, value: str, help_text: str = ""):
    st.markdown(
        f"""<div class="aeg-card aeg-kpi"><div class="aeg-kpi-label" title="{label}">{label}</div>
        <div class="aeg-kpi-value">{value}</div>
        <div class="aeg-kpi-help">{help_text}</div></div>""",
        unsafe_allow_html=True,
    )
