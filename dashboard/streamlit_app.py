"""app.py — Aegis Risk Engine dashboard entry point (thin nav shell)."""
from __future__ import annotations

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

# UI-02: brand block at the top of the sidebar.
# st.navigation always renders FIRST in the sidebar, so anything written with
# `with st.sidebar:` lands BELOW the page links, and st.logo accepts only image
# files (no emoji, no second line). The title and tagline are therefore drawn
# with CSS just above the navigation list. Edit BRAND_TITLE / BRAND_TAGLINE.
BRAND_TITLE = "🛡️ Aegis Risk Engine"
BRAND_TAGLINE = "Explainable insurance-claim fraud scoring · decision support only"

st.markdown(f"""<style>
[data-testid="stSidebarNav"]::before {{
    content: "{BRAND_TITLE}";
    display: block; padding: 0; margin-top: -0.75rem;
    font-size: 1.35rem; font-weight: 700; color: #e6e8ef; line-height: 1.3;
}}
[data-testid="stSidebarNavItems"]::before {{
    content: "{BRAND_TAGLINE}";
    display: block; padding: 0.15rem 0 1rem 0;
    font-size: 0.8rem; color: #9aa1b2; line-height: 1.35;
}}
</style>""", unsafe_allow_html=True)

pg = st.navigation(PAGES)
pg.run()
