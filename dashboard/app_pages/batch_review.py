"""app_pages/batch_review.py — upload a CSV of claims, score them all
in-process, review/escalate high-risk ones."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components.data_access import get_scoring_service, models_are_available
from components.theme import inject_css, page_header, risk_badge

inject_css()
page_header("Batch review queue", "Upload a CSV of claims (any subset of the raw fields) and review the highest-risk ones.")

if not models_are_available():
    st.warning("No trained model found. Run `python -m app.ml.train` from `backend/` first.")
    st.stop()

uploaded = st.file_uploader("Upload claims CSV", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    service = get_scoring_service()
    scored = service.score_batch(df)
    result = pd.concat([df.reset_index(drop=True), scored.reset_index(drop=True)], axis=1)
    st.session_state["batch_result"] = result
    st.session_state["session_scored_count"] = st.session_state.get("session_scored_count", 0) + len(df)

if "batch_result" in st.session_state:
    result = st.session_state["batch_result"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Claims scored", len(result))
    c2.metric("Flagged", int(result["flagged"].sum()))
    c3.metric("High risk", int((result["risk_grade"] == "High").sum()))

    st.write("")
    min_prob = st.slider("Minimum fraud probability to show", 0.0, 1.0, 0.0, 0.05)
    view = result[result["fraud_probability"] >= min_prob].sort_values("fraud_probability", ascending=False)
    st.dataframe(view, use_container_width=True, hide_index=True)

    st.write("")
    st.markdown("#### Escalate a claim for investigation")
    if len(view):
        idx = st.selectbox("Row to escalate", view.index.tolist())
        note = st.text_input("Investigator note")
        if st.button("Escalate"):
            escalated = st.session_state.get("escalated_rows", [])
            escalated.append({**view.loc[idx].to_dict(), "note": note})
            st.session_state["escalated_rows"] = escalated
            st.success(f"Row {idx} escalated for investigation.")

    if st.session_state.get("escalated_rows"):
        st.write("")
        st.markdown("#### Escalated for investigation (this session)")
        esc_df = pd.DataFrame(st.session_state["escalated_rows"])
        st.dataframe(esc_df, use_container_width=True, hide_index=True)
        st.download_button("Download escalated list (CSV)", esc_df.to_csv(index=False), "escalated_claims.csv", "text/csv")
else:
    st.info("Upload a CSV to see results here.")
