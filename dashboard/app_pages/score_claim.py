"""app_pages/score_claim.py — one-off claim scoring form, calls
FraudScoringService IN-PROCESS (same instance the API uses)."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_dashboard_root = str(Path(__file__).resolve().parents[1])
if _dashboard_root not in sys.path:
    # PB-01: append, never insert(0, ...) — inserting the dashboard
    # dir at the FRONT of sys.path on every rerun is what let
    # `import app...` resolve to the old dashboard/app.py instead of
    # the backend's app/ package (see components/data_access.py,
    # which puts backend/ at sys.path[0] once, on first import).
    sys.path.append(_dashboard_root)
from components.data_access import get_scoring_service, models_are_available
from components.theme import inject_css, page_header, risk_badge

inject_css()
page_header("Score a claim", "Fills in the fields this model actually uses — anything left blank falls back to a documented default, same as an external adapter would see.")

if not models_are_available():
    st.warning("No trained model found. Run `python -m app.ml.train` from `backend/` first.")
    st.stop()

with st.form("score_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Insured age", 16, 100, 35)
        months_as_customer = st.number_input("Months as customer", 0, 600, 120)
        policy_deductable = st.selectbox("Policy deductible", [500, 1000, 2000], index=1)
        policy_annual_premium = st.number_input("Annual premium", 0.0, 5000.0, 1300.0)
        witnesses = st.number_input("Witnesses", 0, 5, 1)
    with c2:
        total_claim_amount = st.number_input("Total claim amount", 0.0, 200000.0, 55000.0)
        vehicle_claim = st.number_input("Vehicle claim", 0.0, 150000.0, 35000.0)
        injury_claim = st.number_input("Injury claim", 0.0, 100000.0, 10000.0)
        property_claim = st.number_input("Property claim", 0.0, 100000.0, 10000.0)
        incident_severity = st.selectbox("Incident severity", ["Trivial Damage", "Minor Damage", "Major Damage", "Total Loss"], index=2)
    with c3:
        insured_hobbies = st.text_input("Insured hobby", "reading")
        insured_occupation = st.text_input("Insured occupation", "craft-repair")
        insured_zip = st.number_input("Insured ZIP", 10000, 999999, 468000)
        police_report_available = st.selectbox("Police report available", ["YES", "NO"], index=0)
        auto_year = st.number_input("Auto year", 1990, 2026, 2015)

    submitted = st.form_submit_button("Score claim", type="primary")

if submitted:
    payload = {
        "age": age, "months_as_customer": months_as_customer, "policy_deductable": policy_deductable,
        "policy_annual_premium": policy_annual_premium, "witnesses": witnesses,
        "total_claim_amount": total_claim_amount, "vehicle_claim": vehicle_claim,
        "injury_claim": injury_claim, "property_claim": property_claim,
        "incident_severity": incident_severity, "insured_hobbies": insured_hobbies,
        "insured_occupation": insured_occupation, "insured_zip": insured_zip,
        "police_report_available": police_report_available, "auto_year": auto_year,
        "incident_date": "2024-06-15", "policy_bind_date": "2018-01-01",
    }
    service = get_scoring_service()
    result = service.score_one(payload)
    st.session_state["session_scored_count"] = st.session_state.get("session_scored_count", 0) + 1

    st.write("")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"### {result['fraud_probability']:.1%} fraud probability")
        st.markdown(risk_badge(result["risk_grade"]), unsafe_allow_html=True)
        st.caption(f"Flagged: {'Yes' if result['flagged'] else 'No'} (threshold {result['operating_threshold']:.2f})")
        st.caption(f"Recommended action: {result['recommended_action']}")
    with c2:
        st.markdown("**Top reasons (SHAP)**")
        for r in result["top_reasons"]:
            st.markdown(f"- {r['sentence']}")
