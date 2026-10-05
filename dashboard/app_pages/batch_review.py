"""app_pages/batch_review.py — batch review DASHBOARD (supervisor's top
priority for the defence).

BR-01 (pre-defence rebuild):
* KPI cards: claims scored, flagged for review, high-risk count, value of
  flagged claims, mean fraud probability.
* Charts: risk-grade distribution, probability histogram with the review
  threshold, flag rate by incident severity / incident state, claim amount
  vs. probability, and the factors most often driving flags across the queue.
* Filters (sidebar): risk grade, probability range, flagged-only, incident
  severity, incident state, claim amount range — every KPI, chart and table
  on the page respects them.
* A top-20 most-suspicious-claims table with each claim's top reasons, and a
  per-claim drill-down chart.
* Downloads: full scored batch, current filtered view, top-20, escalations.
* A built-in sample batch so the page can be demonstrated without a file.

BR-02 (bug fix): the old page re-scored AND re-persisted the uploaded file
on EVERY Streamlit rerun — i.e. every time any widget changed. With filters
on the page that would have written duplicate claims to the database on each
click. Scoring + persistence now happen once per distinct file (keyed by a
content hash); filters only re-slice the cached result.

DS-01: decision-support wording throughout — the queue is a list of
RECOMMENDATIONS for human review.
"""
from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_dashboard_root = str(Path(__file__).resolve().parents[1])
if _dashboard_root not in sys.path:
    sys.path.append(_dashboard_root)
from components.charts import (
    amount_vs_probability, driver_frequency_bar, flag_rate_by, grade_distribution_bar,
    probability_histogram, reasons_bar,
)
from components.data_access import SAMPLES_DIR, get_scoring_service, models_are_available, new_db_session
from components.theme import inject_css, kpi_card, page_header

from app.ml.risk_policy import DECISION_SUPPORT_NOTICE

inject_css()
page_header("Batch review dashboard", "Score a batch of claims, see where the risk is, and work the most suspicious first.")
st.markdown(f"<div class='aeg-note'>🧑‍⚖️ {DECISION_SUPPORT_NOTICE}</div>", unsafe_allow_html=True)
st.write("")

if not models_are_available():
    st.warning("No trained model found. Run `python -m app.ml.train` from `backend/` first.")
    st.stop()

SAMPLE_FILE = SAMPLES_DIR / "sample_batch_claims.csv"

# ---------------------------------------------------------------- input ---
c_up, c_sample = st.columns([3, 1])
with c_up:
    uploaded = st.file_uploader("Upload claims CSV (any subset of the raw claim fields)", type=["csv"])
with c_sample:
    st.write("")
    st.write("")
    use_sample = st.button("Load sample batch", width="stretch", disabled=not SAMPLE_FILE.exists(),
                           help="60 demo claims from data/samples/sample_batch_claims.csv")

raw_bytes, source_name = None, None
if uploaded is not None:
    raw_bytes, source_name = uploaded.getvalue(), uploaded.name
elif use_sample:
    raw_bytes, source_name = SAMPLE_FILE.read_bytes(), SAMPLE_FILE.name

if raw_bytes is not None:
    digest = hashlib.sha256(raw_bytes).hexdigest()
    if st.session_state.get("batch_digest") != digest:  # BR-02: once per distinct file
        df = pd.read_csv(io.BytesIO(raw_bytes), keep_default_na=False, na_values=[""])
        service = get_scoring_service()
        with st.spinner(f"Scoring {len(df):,} claims and computing explanations..."):
            scored = service.score_batch(df)
        from app.db.persistence import persist_scored_claims_batch
        scored_rows = [{**row, "model_version": service.model_version} for row in scored.to_dict(orient="records")]
        db = new_db_session()
        try:
            claim_ids = persist_scored_claims_batch(db, df.to_dict(orient="records"), scored_rows, ingested_via="dashboard_batch")
            db.commit()
        finally:
            db.close()
        result = pd.concat([df.reset_index(drop=True), scored.reset_index(drop=True)], axis=1)
        result.insert(0, "claim_id", claim_ids)
        if "total_claim_amount" not in result or result["total_claim_amount"].isna().any():
            parts = result.reindex(columns=["injury_claim", "property_claim", "vehicle_claim"]).apply(pd.to_numeric, errors="coerce")
            derived = parts.sum(axis=1, min_count=3)
            result["total_claim_amount"] = pd.to_numeric(result.get("total_claim_amount"), errors="coerce").fillna(derived)
        result["total_claim_amount"] = pd.to_numeric(result["total_claim_amount"], errors="coerce").fillna(0.0)
        result["top_3_reasons"] = result["top_reasons"].map(
            lambda rs: " | ".join(f"{r['display_name']}: {r['display_value']} ({'▲' if r['shap_value'] > 0 else '▼'})" for r in (rs or [])[:3]))
        st.session_state.update(batch_digest=digest, batch_result=result, batch_source=source_name,
                                session_scored_count=st.session_state.get("session_scored_count", 0) + len(df),
                                escalated_rows=[])
        st.toast(f"Scored and saved {len(claim_ids):,} claims from {source_name}")

if "batch_result" not in st.session_state:
    st.info("Upload a CSV, or click **Load sample batch**, to populate the dashboard.")
    st.stop()

result: pd.DataFrame = st.session_state["batch_result"]
threshold = float(result["operating_threshold"].iloc[0])

# -------------------------------------------------------------- filters ---
with st.sidebar:
    st.markdown("#### Batch filters")
    grades = st.multiselect("Risk grade", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    p_lo, p_hi = st.slider("Fraud probability", 0.0, 1.0, (0.0, 1.0), 0.01)
    flagged_only = st.checkbox("Flagged for review only")
    sev_opts = sorted(result["incident_severity"].dropna().astype(str).unique()) if "incident_severity" in result else []
    severities = st.multiselect("Incident severity", sev_opts, default=sev_opts)
    state_opts = sorted(result["incident_state"].dropna().astype(str).unique()) if "incident_state" in result else []
    states = st.multiselect("Incident state", state_opts, default=state_opts)
    amt_max = float(max(1.0, result["total_claim_amount"].max()))
    a_lo, a_hi = st.slider("Total claim amount ($)", 0.0, amt_max, (0.0, amt_max), step=float(max(1.0, round(amt_max / 100))))
    if st.button("Clear batch"):
        for k in ("batch_digest", "batch_result", "batch_source", "escalated_rows"):
            st.session_state.pop(k, None)
        st.rerun()

mask = (result["risk_grade"].isin(grades) & result["fraud_probability"].between(p_lo, p_hi)
        & result["total_claim_amount"].between(a_lo, a_hi))
if flagged_only:
    mask &= result["flagged"]
if sev_opts:
    mask &= result["incident_severity"].astype(str).isin(severities)
if state_opts:
    mask &= result["incident_state"].astype(str).isin(states)
view = result[mask].sort_values("fraud_probability", ascending=False)

st.caption(f"Source: **{st.session_state.get('batch_source')}** · showing **{len(view):,}** of {len(result):,} claims "
           f"after filters · review threshold {threshold:.2f}")

# ----------------------------------------------------------------- KPIs ---
flagged = view[view["flagged"]]
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    kpi_card("Claims in view", f"{len(view):,}", f"of {len(result):,} scored")
with k2:
    kpi_card("Recommended for review", f"{len(flagged):,}", f"{len(flagged) / max(1, len(view)):.0%} of view · ≥ {threshold:.2f}")
with k3:
    kpi_card("High risk", f"{int((view['risk_grade'] == 'High').sum()):,}", "priority SIU review suggested")
with k4:
    kpi_card("Value under review", f"${flagged['total_claim_amount'].sum():,.0f}",
             f"{flagged['total_claim_amount'].sum() / max(1.0, view['total_claim_amount'].sum()):.0%} of view value")
with k5:
    kpi_card("Mean fraud probability", f"{view['fraud_probability'].mean():.1%}" if len(view) else "—", "across claims in view")

if view.empty:
    st.warning("No claims match the current filters.")
    st.stop()

# --------------------------------------------------------------- charts ---
t_overview, t_drivers, t_top, t_all = st.tabs(["📊 Risk overview", "🔎 What drives the flags", "🚩 Top 20 suspicious", "📋 All claims"])

with t_overview:
    c1, c2 = st.columns(2)
    c1.plotly_chart(grade_distribution_bar(view), width="stretch")
    c2.plotly_chart(probability_histogram(view, threshold), width="stretch")
    c3, c4 = st.columns(2)
    if "incident_severity" in view:
        c3.plotly_chart(flag_rate_by(view, "incident_severity", "Review rate by incident severity"), width="stretch")
    c4.plotly_chart(amount_vs_probability(view, threshold), width="stretch")
    if "incident_state" in view:
        st.plotly_chart(flag_rate_by(view, "incident_state", "Review rate by incident state"), width="stretch")

with t_drivers:
    if len(flagged):
        st.plotly_chart(driver_frequency_bar(flagged["top_reasons"].tolist()), width="stretch")
        st.caption("Counts how often each claim field appears among the top-3 risk-RAISING factors of a flagged claim. "
                   "If one field dominates, the queue is effectively being driven by that field — worth knowing before "
                   "trusting the ranking (see the Model insights page on the Major-Damage baseline).")
    else:
        st.info("No flagged claims in the current view.")

top20 = view.head(20).copy()
top20.insert(0, "rank", range(1, len(top20) + 1))
TOP_COLS = [c for c in ["rank", "claim_id", "fraud_probability", "risk_grade", "recommended_action", "total_claim_amount",
                        "incident_severity", "incident_state", "police_report_available", "witnesses", "top_3_reasons"]
            if c in top20.columns]

with t_top:
    st.dataframe(
        top20[TOP_COLS], width="stretch", hide_index=True,
        column_config={
            "fraud_probability": st.column_config.ProgressColumn("Fraud probability", format="%.2f", min_value=0.0, max_value=1.0),
            "total_claim_amount": st.column_config.NumberColumn("Total claim", format="$%d"),
            "top_3_reasons": st.column_config.TextColumn("Top 3 reasons (▲ raises / ▼ lowers)", width="large"),
        },
    )
    pick = st.selectbox("Explain a claim", top20["claim_id"].tolist(),
                        format_func=lambda cid: f"Claim #{cid} — p={float(top20.loc[top20['claim_id'] == cid, 'fraud_probability'].iloc[0]):.2f}")
    row = top20[top20["claim_id"] == pick].iloc[0]
    st.plotly_chart(reasons_bar(row["top_reasons"], title=f"Why claim #{pick} scored {row['fraud_probability']:.0%}"),
                    width="stretch")
    st.markdown(f"**Suggested next step:** {row['recommended_action']}")

    with st.form("escalate"):
        st.markdown("##### Escalate this claim for investigation")
        investigator_name = st.text_input("Investigator name")
        note = st.text_input("Investigator note")
        if st.form_submit_button("Escalate claim"):
            from app.db.persistence import persist_escalation
            db = new_db_session()
            try:
                persist_escalation(db, int(pick), investigator_name, note)
                db.commit()
            finally:
                db.close()
            st.session_state.setdefault("escalated_rows", []).append(
                {**row.drop(labels=["top_reasons"]).to_dict(), "investigator": investigator_name, "note": note})
            st.success(f"Claim #{pick} escalated for investigation and recorded in the audit log.")

with t_all:
    show = view.drop(columns=["top_reasons"])
    st.dataframe(show, width="stretch", hide_index=True,
                 column_config={"fraud_probability": st.column_config.ProgressColumn("Fraud probability", format="%.2f",
                                                                                    min_value=0.0, max_value=1.0)})

# ------------------------------------------------------------ downloads ---
st.write("")
st.markdown("#### Downloads")
d1, d2, d3, d4 = st.columns(4)
export_all = result.drop(columns=["top_reasons"])
d1.download_button("⬇ Full scored batch", export_all.to_csv(index=False), "aegis_scored_batch.csv", "text/csv", width="stretch")
d2.download_button("⬇ Current filtered view", view.drop(columns=["top_reasons"]).to_csv(index=False), "aegis_filtered_view.csv",
                   "text/csv", width="stretch")
d3.download_button("⬇ Top 20 suspicious", top20[TOP_COLS].to_csv(index=False), "aegis_top20_suspicious.csv", "text/csv",
                   width="stretch")
esc = pd.DataFrame(st.session_state.get("escalated_rows", []))
d4.download_button("⬇ Escalated claims", esc.to_csv(index=False) if len(esc) else "no escalations yet\n",
                   "aegis_escalated_claims.csv", "text/csv", width="stretch", disabled=esc.empty)
