"""app_pages/batch_review.py — Batch review dashboard (supervisor priority).

A1  KPI cards (total, flagged, High/Medium/Low counts, mean fraud-risk score);
    charts: risk bands, score histogram, risk by incident severity,
    aggregated SHAP drivers (mean |SHAP| per PARENT raw field), claim amount
    vs score; then an auto "Top 20 suspicious claims" table. The full table
    is secondary (expander).
A2  Sidebar filters (band, flagged, score range, severity, claim amount)
    drive BOTH the charts and the tables. No click-on-chart filtering.
A3  Drill-down on the same page: select a row in the top-20 table (or pick
    it from the list below it) to open its full explanation.
A4  Downloads: filtered queue and escalated list (CSV), plus rejected rows.
A6  Rows whose total != injury + property + vehicle (tolerance 1.0) are
    REJECTED with a per-row reason, never silently corrected or scored.
BR-02 (kept) scoring + persistence happen once per distinct file.
"""
from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_root = str(Path(__file__).resolve().parents[1])
if _root not in sys.path:
    sys.path.append(_root)
from components.charts import (amount_vs_probability, field_driver_bar,
                               probability_histogram, reasons_bar, score_by_category)
from components.data_access import SAMPLES_DIR, get_scoring_service, models_are_available, new_db_session
from components.theme import inject_css, kpi_card, page_header

from app.ml.explainer import FEATURE_LABELS, FEATURE_LABELS_EXTRA, split_reasons
from app.ml.feature_engineering import TOTAL_TOLERANCE, inconsistent_total_mask
from app.ml.risk_policy import DECISION_SUPPORT_NOTICE

inject_css()
page_header("Batch review dashboard", "Score a batch of claims, see where the risk is, and open the most suspicious first.")
st.markdown(f"<div class='aeg-note'>🧑‍⚖️ {DECISION_SUPPORT_NOTICE}</div>", unsafe_allow_html=True)
st.write("")
if not models_are_available():
    st.warning("No trained model found. Run `python -m app.ml.run_all` from `backend/` first.")
    st.stop()

SAMPLE_FILE = SAMPLES_DIR / "sample_batch_claims.csv"
LABELS = {**FEATURE_LABELS, **FEATURE_LABELS_EXTRA}

c_up, c_sample = st.columns([3, 1])
uploaded = c_up.file_uploader("Upload claims CSV (any subset of the raw claim fields)", type=["csv"])
with c_sample:
    st.write(""); st.write("")
    use_sample = st.button("Load sample batch", width="stretch", disabled=not SAMPLE_FILE.exists())

raw_bytes, source = (uploaded.getvalue(), uploaded.name) if uploaded is not None else \
    ((SAMPLE_FILE.read_bytes(), SAMPLE_FILE.name) if use_sample else (None, None))

if raw_bytes is not None:
    digest = hashlib.sha256(raw_bytes).hexdigest()
    if st.session_state.get("batch_digest") != digest:
        df = pd.read_csv(io.BytesIO(raw_bytes), keep_default_na=False, na_values=[""])
        bad = inconsistent_total_mask(df)
        rejected = df[bad].copy()
        if len(rejected):
            parts = rejected[["injury_claim", "property_claim", "vehicle_claim"]].apply(pd.to_numeric, errors="coerce").sum(axis=1)
            rejected.insert(0, "rejection_reason", [f"total_claim_amount {t:,.2f} != sum of parts {s:,.2f} (tolerance {TOTAL_TOLERANCE})"
                                                     for t, s in zip(pd.to_numeric(rejected["total_claim_amount"]), parts)])
            rejected.insert(0, "row_number", rejected.index + 1)
        df = df[~bad].reset_index(drop=True)
        service = get_scoring_service()
        with st.spinner(f"Scoring {len(df):,} claims and computing explanations..."):
            scored, field_shap = service.score_batch(df, with_field_shap=True) if len(df) else (pd.DataFrame(), pd.DataFrame())
        claim_ids = []
        if len(df):
            from app.db.persistence import persist_scored_claims_batch
            rows = [{**r, "model_version": service.model_version} for r in scored.to_dict(orient="records")]
            db = new_db_session()
            try:
                claim_ids = persist_scored_claims_batch(db, df.to_dict(orient="records"), rows, ingested_via="dashboard_batch")
                db.commit()
            finally:
                db.close()
        result = pd.concat([df, scored.reset_index(drop=True)], axis=1)
        result.insert(0, "claim_id", claim_ids)
        if "total_claim_amount" not in result:
            result["total_claim_amount"] = result.reindex(columns=["injury_claim", "property_claim", "vehicle_claim"]).sum(axis=1)
        result["total_claim_amount"] = pd.to_numeric(result["total_claim_amount"], errors="coerce").fillna(
            result.reindex(columns=["injury_claim", "property_claim", "vehicle_claim"]).apply(pd.to_numeric, errors="coerce").sum(axis=1))
        result["top_3_reasons"] = result["top_reasons"].map(
            lambda rs: " | ".join(f"{r['display_name']}: {r['display_value']} ({'▲' if r['shap_value'] > 0 else '▼'})" for r in (rs or [])[:3]))
        field_shap.index = result.index
        st.session_state.update(batch_digest=digest, batch_result=result, batch_field_shap=field_shap,
                                batch_rejected=rejected, batch_source=source, escalated_rows=[])

if "batch_result" not in st.session_state:
    st.info("Upload a CSV, or click **Load sample batch**, to populate the dashboard.")
    st.stop()

result: pd.DataFrame = st.session_state["batch_result"]
rejected: pd.DataFrame = st.session_state["batch_rejected"]
if len(rejected):
    st.error(f"{len(rejected)} row(s) rejected: total_claim_amount does not equal injury + property + vehicle. "
             "They were not scored. See the rejected-rows report below.")
if result.empty:
    st.stop()
threshold = float(result["operating_threshold"].iloc[0])

# ------------------------------------------------------------- filters (A2)
with st.sidebar:
    st.markdown("#### Batch filters")
    bands = st.multiselect("Risk band", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    flagged_only = st.checkbox("Flagged for review only")
    s_lo, s_hi = st.slider("Fraud-risk score", 0.0, 1.0, (0.0, 1.0), 0.01)
    sev_opts = sorted(result["incident_severity"].dropna().astype(str).unique()) if "incident_severity" in result else []
    sev = st.multiselect("Incident severity", sev_opts, default=sev_opts)
    amax = float(max(1.0, result["total_claim_amount"].max()))
    a_lo, a_hi = st.slider("Total claim amount ($)", 0.0, amax, (0.0, amax), step=float(max(1.0, round(amax / 100))))
    if st.button("Clear batch"):
        for k in [k for k in st.session_state if k.startswith("batch_")] + ["escalated_rows"]:
            st.session_state.pop(k, None)
        st.rerun()

mask = (result["risk_grade"].isin(bands) & result["fraud_probability"].between(s_lo, s_hi)
        & result["total_claim_amount"].between(a_lo, a_hi))
if flagged_only:
    mask &= result["flagged"]
if sev_opts:
    mask &= result["incident_severity"].astype(str).isin(sev)
view = result[mask].sort_values("fraud_probability", ascending=False)
st.caption(f"Source **{st.session_state.get('batch_source')}** · {len(view):,} of {len(result):,} scored claims match the filters · "
           f"review threshold {threshold:.2f}")

# --------------------------------------------------------------- KPIs (A1)
k = st.columns(6)
cards = [("Claims in view", f"{len(view):,}", f"of {len(result):,} scored"),
         ("Flagged", f"{int(view['flagged'].sum()):,}", f"for review · score ≥ {threshold:.2f}"),
         ("High", f"{int((view['risk_grade'] == 'High').sum()):,}", "risk band"),
         ("Medium", f"{int((view['risk_grade'] == 'Medium').sum()):,}", "risk band"),
         ("Low", f"{int((view['risk_grade'] == 'Low').sum()):,}", "risk band"),
         ("Mean score", f"{view['fraud_probability'].mean():.2f}" if len(view) else "—", "fraud-risk score")]
for col, (t, v, sub) in zip(k, cards):
    with col:
        kpi_card(t, v, sub)
if view.empty:
    st.warning("No claims match the current filters.")
    st.stop()

# ------------------------------------------------------------- charts (A1)
# UI-15: no risk-grade bar (it duplicated the KPI cards). Histogram carries the
# band edges and the review threshold; the scatter is smaller and last.
st.plotly_chart(probability_histogram(view, threshold, get_scoring_service().band_edges), width="stretch")
c3, c4 = st.columns(2)
if "incident_severity" in view:
    c3.plotly_chart(score_by_category(view, "incident_severity", "Mean fraud-risk score by incident severity"), width="stretch")
c4.plotly_chart(field_driver_bar(st.session_state["batch_field_shap"].loc[view.index], LABELS, top_n=8), width="stretch")
st.plotly_chart(amount_vs_probability(view, threshold), width="stretch")

# ---------------------------------------------- top 20 + drill-down (A3)
st.markdown("#### Top 20 suspicious claims")
top20 = view.head(20).copy()
top20.insert(0, "rank", range(1, len(top20) + 1))
TOP_COLS = [c for c in ["rank", "claim_id", "fraud_probability", "risk_grade", "flagged", "recommended_action",
                        "total_claim_amount", "incident_severity", "top_3_reasons"] if c in top20.columns]
event = st.dataframe(
    top20[TOP_COLS], width="stretch", hide_index=True, key="top20_table", on_select="rerun", selection_mode="single-row",
    column_config={"fraud_probability": st.column_config.ProgressColumn("Fraud-risk score", format="%.2f", min_value=0.0, max_value=1.0),
                   "total_claim_amount": st.column_config.NumberColumn("Total claim", format="$%d"),
                   "top_3_reasons": st.column_config.TextColumn("Top 3 reasons (▲ raises / ▼ lowers)", width="large")})
selected_rows = getattr(getattr(event, "selection", None), "rows", []) or []
ids = top20["claim_id"].tolist()
default_idx = selected_rows[0] if selected_rows else 0
pick = st.selectbox("Claim to explain (or select a row above)", ids, index=default_idx,
                    format_func=lambda cid: f"Claim #{cid}")
row = top20[top20["claim_id"] == pick].iloc[0]

with st.container(border=True):
    st.markdown(f"##### Claim #{pick} — fraud-risk score {row['fraud_probability']:.2f} · {row['risk_grade']} band · "
                f"{'flagged for review' if row['flagged'] else 'not flagged'}")
    st.markdown(f"**Recommended next step:** {row['recommended_action']}")
    up, down = split_reasons(row["top_reasons"], k=3)  # UI-15: top 3 each
    a, b = st.columns(2)
    a.markdown("**Factors increasing risk**\n" + ("\n".join(f"- {r['sentence']}" for r in up) or "- none"))
    b.markdown("**Factors reducing risk**\n" + ("\n".join(f"- {r['sentence']}" for r in down) or "- none"))
    st.plotly_chart(reasons_bar(row["top_reasons"], title=f"Contributions for claim #{pick}"), width="stretch")
    with st.form("escalate"):
        investigator = st.text_input("Investigator name")
        note = st.text_input("Investigator note")
        if st.form_submit_button("Escalate claim"):
            from app.db.persistence import persist_escalation
            db = new_db_session()
            try:
                persist_escalation(db, int(pick), investigator, note)
                db.commit()
            finally:
                db.close()
            st.session_state.setdefault("escalated_rows", []).append(
                {**row.drop(labels=["top_reasons"]).to_dict(), "investigator": investigator, "note": note})
            st.success(f"Claim #{pick} escalated for investigation and recorded in the audit log.")

with st.expander(f"Full filtered queue ({len(view):,} claims)"):
    st.dataframe(view.drop(columns=["top_reasons"]), width="stretch", hide_index=True)
if len(rejected):
    with st.expander(f"Rejected rows ({len(rejected)})", expanded=True):
        st.dataframe(rejected, width="stretch", hide_index=True)

# ------------------------------------------------------------ downloads (A4)
st.markdown("#### Downloads")
d1, d2, d3 = st.columns(3)
d1.download_button("⬇ Filtered queue (CSV)", view.drop(columns=["top_reasons"]).to_csv(index=False),
                   "aegis_filtered_queue.csv", "text/csv", width="stretch")
esc = pd.DataFrame(st.session_state.get("escalated_rows", []))
d2.download_button("⬇ Escalated list (CSV)", esc.to_csv(index=False) if len(esc) else "no escalations yet\n",
                   "aegis_escalated.csv", "text/csv", width="stretch", disabled=esc.empty)
d3.download_button("⬇ Rejected rows (CSV)", rejected.to_csv(index=False) if len(rejected) else "none\n",
                   "aegis_rejected_rows.csv", "text/csv", width="stretch", disabled=rejected.empty)
