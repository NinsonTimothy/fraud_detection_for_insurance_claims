"""app_pages/batch_review.py — upload a CSV of claims, score them all
in-process, review/escalate high-risk ones.

PB-12 (fixed): both the scored batch and any "escalated for
investigation" rows used to live only in `st.session_state` — gone on
refresh, never visible to the API, the audit log, or another analyst.
Now persisted through `db/persistence.py`'s `persist_scored_claims_batch()`/
`persist_escalation()`."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_dashboard_root = str(Path(__file__).resolve().parents[1])
if _dashboard_root not in sys.path:
    # PB-01: append, never insert(0, ...) — inserting the dashboard
    # dir at the FRONT of sys.path on every rerun is what let
    # `import app...` resolve to the old dashboard/app.py instead of
    # the backend's app/ package (see components/data_access.py,
    # which puts backend/ at sys.path[0] once, on first import).
    sys.path.append(_dashboard_root)
from components.data_access import get_scoring_service, models_are_available, new_db_session
from components.theme import inject_css, page_header, risk_badge

inject_css()
page_header("Batch review queue", "Upload a CSV of claims (any subset of the raw fields) and review the highest-risk ones.")

if not models_are_available():
    st.warning("No trained model found. Run `python -m app.ml.train` from `backend/` first.")
    st.stop()

uploaded = st.file_uploader("Upload claims CSV", type=["csv"])
if uploaded is not None:
    # SH-01: keep_default_na=False so authorities_contacted's genuine
    # "None" category isn't misread as missing data by pandas' default
    # NA-sentinel list — see backend/app/ml/clean_data.py's docstring.
    df = pd.read_csv(uploaded, keep_default_na=False, na_values=[""])
    service = get_scoring_service()
    scored = service.score_batch(df)
    result = pd.concat([df.reset_index(drop=True), scored.reset_index(drop=True)], axis=1)

    # PB-12: persist the whole batch, same as api/scoring.py's POST
    # /score/batch does — score_batch() never computes top_reasons (no
    # SHAP call for batch scoring) and doesn't return model_version, so
    # both are filled in here the same way scoring.py already does.
    from app.db.persistence import persist_scored_claims_batch
    raw_rows = df.to_dict(orient="records")
    scored_rows = [
        {**row, "model_version": service.model_version, "top_reasons": None}
        for row in scored.to_dict(orient="records")
    ]
    db = new_db_session()
    try:
        claim_ids = persist_scored_claims_batch(db, raw_rows, scored_rows, ingested_via="dashboard_batch")
        db.commit()
    finally:
        db.close()
    result.insert(0, "claim_id", claim_ids)

    st.session_state["batch_result"] = result
    st.session_state["session_scored_count"] = st.session_state.get("session_scored_count", 0) + len(df)
    st.caption(f"Persisted {len(claim_ids)} claim(s) — visible via the API's /claims and the audit log.")

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
        investigator_name = st.text_input("Investigator name")
        note = st.text_input("Investigator note")
        if st.button("Escalate"):
            row = view.loc[idx].to_dict()

            # PB-12: "Escalate" used to only append to st.session_state —
            # gone on refresh, invisible to the API/audit log/another
            # analyst. Now also recorded as a claim_escalated audit-log
            # entry against the claim's real DB id (persist_escalation()
            # deliberately does NOT create an InvestigatorFeedback row —
            # see db/persistence.py's docstring for why "escalate" and
            # "feedback" are kept as two different event types).
            from app.db.persistence import persist_escalation
            db = new_db_session()
            try:
                persist_escalation(db, int(row["claim_id"]), investigator_name, note)
                db.commit()
            finally:
                db.close()

            escalated = st.session_state.get("escalated_rows", [])
            escalated.append({**row, "note": note})
            st.session_state["escalated_rows"] = escalated
            st.success(f"Row {idx} (claim #{row['claim_id']}) escalated for investigation.")

    if st.session_state.get("escalated_rows"):
        st.write("")
        st.markdown("#### Escalated for investigation (this session)")
        esc_df = pd.DataFrame(st.session_state["escalated_rows"])
        st.dataframe(esc_df, use_container_width=True, hide_index=True)
        st.download_button("Download escalated list (CSV)", esc_df.to_csv(index=False), "escalated_claims.csv", "text/csv")
else:
    st.info("Upload a CSV to see results here.")
