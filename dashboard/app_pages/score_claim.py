"""app_pages/score_claim.py — one-off claim scoring form, calls
FraudScoringService IN-PROCESS (same instance the API uses).

PB-12 (fixed): the scored result used to be shown and then discarded —
nothing was written to the DB, so a claim scored here never showed up in
`GET /claims`, the risk grid, or the audit log the way an API-scored
claim did. Now persisted through `db/persistence.py`'s
`persist_scored_claim()`, the exact same function `api/scoring.py`'s
`POST /score` uses, tagged `ingested_via="dashboard"` so the audit trail
can tell the two surfaces apart.

PB-19 (fixed): `incident_date`/`policy_bind_date` used to be hardcoded
("2024-06-15"/"2018-01-01") for EVERY claim scored here, regardless of
`months_as_customer` or anything else the analyst entered — silently
wrong for any claim that wasn't meant to describe exactly that policy
age. Reproduced directly: scoring the same claim with `months_as_customer`
=6 through the always-hardcoded dates vs. through dates actually
consistent with a 6-month-old policy changed `fraud_probability` by
~0.003 — small on this dataset, but a real, silent effect on
`policy_age_days`/`vehicle_age_at_incident` (feature_engineering.py), not
a cosmetic one. Both are now real `st.date_input` widgets.
`insured_hobbies`/`insured_occupation` used to be free `st.text_input`
fields with no indication of the schema's actual (fixed, 20/14-value)
vocabulary — harmless for the shipped model's default config (see the
caption below), but a real risk if `INCLUDE_PROXY_FEATURES=true`
(SH-02/D3) is ever toggled on: a typo silently fails the `isin()`/`==`
membership checks feature_engineering.py builds `is_highrisk_hobby`/
`is_exec_occupation` from. Now `st.selectbox`es over
`feature_engineering.KNOWN_HOBBIES`/`KNOWN_OCCUPATIONS`, the same
constants that vocabulary check is actually built from — can't drift
apart from what the model layer recognizes."""
from __future__ import annotations

import sys
from datetime import date, timedelta
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
from components.data_access import get_scoring_service, models_are_available, new_db_session
from components.theme import inject_css, page_header, risk_badge

from app.core.config import INCLUDE_PROXY_FEATURES
from app.ml.feature_engineering import KNOWN_HOBBIES, KNOWN_OCCUPATIONS

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
        insured_hobbies = st.selectbox("Insured hobby", KNOWN_HOBBIES, index=KNOWN_HOBBIES.index("reading"))
        insured_occupation = st.selectbox("Insured occupation", KNOWN_OCCUPATIONS, index=KNOWN_OCCUPATIONS.index("craft-repair"))
        insured_zip = st.number_input("Insured ZIP", 10000, 999999, 468000)
        police_report_available = st.selectbox("Police report available", ["YES", "NO"], index=0)
        auto_year = st.number_input("Auto year", 1990, 2026, 2015)
        # PB-19: default bind date consistent with months_as_customer
        # above (~30.44 days/month) rather than a fixed constant that
        # ignores it — still freely editable, this is just a starting
        # point that isn't already self-contradictory. Explicit `key=` on
        # both date_inputs: without one, Streamlit derives a widget's
        # identity partly from its own call arguments, and incident_date's
        # `min_value` below is itself derived from policy_bind_date's live
        # value — so editing policy_bind_date on one rerun silently
        # resets incident_date back to its coded default on the next,
        # discarding whatever the analyst had entered there. Reproduced
        # directly against a bare AppTest session before adding `key=`:
        # setting Incident date, then changing Policy bind date, silently
        # reverted Incident date to today(). A fixed `key=` keeps each
        # widget's own state independent of the other's current value.
        policy_bind_date = st.date_input(
            "Policy bind date", value=date.today() - timedelta(days=round(months_as_customer * 30.44)),
            min_value=date(1990, 1, 1), max_value=date.today(), key="policy_bind_date",
        )
        # feature_engineering.py's own comment flags incident-before-bind
        # as a data ISSUE (a negative "policy age"), so min_value enforces
        # incident_date >= policy_bind_date here rather than letting the
        # form build another instance of it.
        incident_date = st.date_input("Incident date", value=date.today(), min_value=policy_bind_date, max_value=date.today(), key="incident_date")

    if not INCLUDE_PROXY_FEATURES:
        st.caption("Hobby/occupation don't affect this score in the shipped default configuration — `INCLUDE_PROXY_FEATURES` is off (SH-02/D3, see docs/REBUILD_NOTES.md). Still validated against the real category schema below.")

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
        "incident_date": incident_date.isoformat(), "policy_bind_date": policy_bind_date.isoformat(),
    }
    service = get_scoring_service()
    result = service.score_one(payload)
    st.session_state["session_scored_count"] = st.session_state.get("session_scored_count", 0) + 1

    # PB-12: persist, same as api/scoring.py's POST /score does.
    from app.db.persistence import persist_scored_claim
    db = new_db_session()
    try:
        claim = persist_scored_claim(db, payload, result, ingested_via="dashboard")
        db.commit()
        claim_id = claim.id
    finally:
        db.close()

    st.write("")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"### {result['fraud_probability']:.1%} fraud probability")
        st.markdown(risk_badge(result["risk_grade"]), unsafe_allow_html=True)
        st.caption(f"Flagged: {'Yes' if result['flagged'] else 'No'} (threshold {result['operating_threshold']:.2f})")
        st.caption(f"Recommended action: {result['recommended_action']}")
        st.caption(f"Saved as claim #{claim_id} — visible via the API's /claims/{claim_id} and the audit log.")
    with c2:
        st.markdown("**Top reasons (SHAP)**")
        for r in result["top_reasons"]:
            st.markdown(f"- {r['sentence']}")
