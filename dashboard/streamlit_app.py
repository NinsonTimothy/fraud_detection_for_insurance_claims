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
        st.Page("app_pages/batch_review.py", title="Batch review queue", url_path="batch"),
    ],
    "Score": [
        st.Page("app_pages/score_claim.py", title="Score a claim", url_path="score"),
    ],
    "Model": [
        st.Page("app_pages/model_insights.py", title="Model insights", url_path="insights"),
    ],
}

pg = st.navigation(PAGES)
with st.sidebar:
    st.markdown("### 🛡️ Aegis Risk Engine")
    st.caption("Explainable insurance-claim fraud scoring")
pg.run()
