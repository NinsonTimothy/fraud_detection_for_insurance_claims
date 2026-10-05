"""app.py — Aegis Risk Engine dashboard entry point (thin nav shell)."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Aegis Risk Engine", page_icon="🛡️", layout="wide")

PAGES = {
    "Monitor": [
        st.Page("app_pages/overview.py", title="Overview", url_path="", default=True),
        st.Page("app_pages/monitoring.py", title="Monitoring & external validation", url_path="monitoring"),
    ],
    "Investigate": [
        st.Page("app_pages/batch_review.py", title="Batch review dashboard", url_path="batch"),
    ],
    "Score": [
        st.Page("app_pages/score_claim.py", title="Score a claim", url_path="score"),
    ],
    "Model": [
        st.Page("app_pages/model_insights.py", title="Model insights", url_path="insights"),
    ],
}

# UI-02: st.navigation always renders at the TOP of the sidebar, so a title
# written into the sidebar lands BELOW the page links. st.logo is pinned above
# the navigation (and stays visible in the header when the sidebar collapses).
_ASSETS = Path(__file__).resolve().parent / "assets"
st.logo(str(_ASSETS / "aegis_wordmark.svg"), size="large", icon_image=str(_ASSETS / "aegis_icon.svg"))

pg = st.navigation(PAGES)
with st.sidebar:
    st.caption("Explainable insurance-claim fraud scoring · decision support only")
pg.run()
