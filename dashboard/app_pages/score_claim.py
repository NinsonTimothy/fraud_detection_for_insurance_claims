"""app_pages/score_claim.py — score one claim and SEE why.

Pre-defence changes (supervisor feedback):

SC-01  The form used to collect only ~15 fields. Fields carrying roughly a
       fifth of the model's SHAP weight (incident state, umbrella limit,
       bodily injuries, collision type, authorities contacted, ...) were
       never asked for, so they silently fell back to defaults on every
       claim scored here. Meanwhile it DID ask for hobby, occupation and ZIP,
       which the shipped model does not use at all (proxy features are off;
       ZIP was removed in PB-02). The form is now built from the fields the
       model actually uses, and the page shows what share of the model's
       weight the form covers, computed from shap_importance_by_field.csv.
TC-01  Total claim amount is no longer typed: it is injury + property +
       vehicle, shown live, exactly as in every training row.
VZ-01  The result is visual: a risk gauge with the review threshold, a
       contribution chart (one bar per claim field, labelled with this
       claim's real value — see the EX-01 police-report fix in
       explainer.py), where the claim sits among the 200 test claims, and the
       claim-amount breakdown.
DS-01  Decision-support wording: the system recommends; a person decides.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

_dashboard_root = str(Path(__file__).resolve().parents[1])
if _dashboard_root not in sys.path:
    sys.path.append(_dashboard_root)
from components.charts import reasons_bar
from components.data_access import (
    PROCESSED_DIR, get_scoring_service, load_field_importance, models_are_available, new_db_session,
)
from components.theme import DANGER, MUTED, SUCCESS, WARNING, inject_css, page_header, risk_badge

from app.core.config import INCLUDE_PROXY_FEATURES
from app.ml.explainer import build_source_map
from app.ml.feature_engineering import KNOWN_HOBBIES, KNOWN_OCCUPATIONS, MISSING_COLUMN_DEFAULTS
from app.ml.risk_policy import DECISION_SUPPORT_NOTICE, HIGH_RISK_EDGE, MEDIUM_RISK_EDGE

inject_css()
page_header("Score a claim", "Enter a claim, get a fraud-risk score and a visual explanation of what drove it.")
st.markdown(f"<div class='aeg-note'>🧑‍⚖️ {DECISION_SUPPORT_NOTICE}</div>", unsafe_allow_html=True)
st.write("")

if not models_are_available():
    st.warning("No trained model found. Run `python -m app.ml.train` from `backend/` first.")
    st.stop()

service = get_scoring_service()
D = MISSING_COLUMN_DEFAULTS

# Category options come from the trained model's own one-hot columns, so the
# form can never offer a value the model has not seen (or miss one it has).
_known: dict[str, list[str]] = {}
for _col, (_src, _cat) in build_source_map(service.feature_columns).items():
    if _cat is not None:  # derived single-parent features carry no category
        _known.setdefault(_src, []).append(_cat)
_known = {k: sorted(v) for k, v in _known.items()}


def _cat(label: str, field: str, help_text: str | None = None):
    opts = _known.get(field, [str(D[field])])
    default = str(D[field])
    return st.selectbox(label, opts, index=opts.index(default) if default in opts else 0, key=f"f_{field}", help=help_text)


# Raw fields that feed each engineered feature, for the coverage check.
ENGINEERED_INPUTS = {
    "claim_to_premium_ratio": {"injury_claim", "property_claim", "vehicle_claim", "policy_annual_premium"},
    "vehicle_claim_pct": {"injury_claim", "property_claim", "vehicle_claim"},
    "injury_claim_pct": {"injury_claim", "property_claim", "vehicle_claim"},
    "property_claim_pct": {"injury_claim", "property_claim", "vehicle_claim"},
    "total_claim_amount": {"injury_claim", "property_claim", "vehicle_claim"},
    "policy_age_at_incident_days": {"policy_bind_date", "incident_date"},
    "is_new_customer": {"months_as_customer"},
    "vehicle_age_at_incident": {"incident_date", "auto_year"},
    "incident_severity_ordinal": {"incident_severity"},
    "is_major_damage": {"incident_severity"},
    "is_highrisk_hobby": {"insured_hobbies"},
    "is_exec_occupation": {"insured_occupation"},
}

tab_policy, tab_person, tab_incident, tab_amounts = st.tabs(
    ["① Policy", "② Insured person", "③ Incident", "④ Claim amounts & vehicle"])

with tab_policy:
    c1, c2, c3 = st.columns(3)
    with c1:
        months_as_customer = st.number_input("Months as customer", 0, 600, 120, key="f_months")
        policy_state = _cat("Policy state", "policy_state")
    with c2:
        policy_csl = _cat("Policy CSL (combined single limit)", "policy_csl")
        policy_deductable = st.selectbox("Policy deductible ($)", [500, 1000, 2000], index=1, key="f_ded")
    with c3:
        policy_annual_premium = st.number_input("Annual premium ($)", 0.0, 5000.0, 1256.0, step=50.0, key="f_prem")
        umbrella_limit = st.select_slider(
            "Umbrella limit ($)", options=[0, 1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000,
                                           6_000_000, 7_000_000, 8_000_000, 9_000_000, 10_000_000],
            value=0, format_func=lambda v: f"{v/1e6:.0f}M" if v else "None", key="f_umb")
    policy_bind_date = st.date_input(
        "Policy bind date", value=date.today() - timedelta(days=round(months_as_customer * 30.44)),
        min_value=date(1990, 1, 1), max_value=date.today(), key="policy_bind_date")

with tab_person:
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Insured age", 16, 100, 39, key="f_age")
        insured_sex = _cat("Insured sex", "insured_sex")
    with c2:
        insured_education_level = _cat("Education level", "insured_education_level")
        insured_relationship = _cat("Relationship to policyholder", "insured_relationship")
    with c3:
        capital_gains = st.number_input("Capital gains ($)", 0, 200_000, 0, step=1000, key="f_cg")
        capital_loss = st.number_input("Capital loss ($, enter as negative)", -200_000, 0, 0, step=1000, key="f_cl")
    if INCLUDE_PROXY_FEATURES:
        c4, c5 = st.columns(2)
        insured_hobbies = c4.selectbox("Insured hobby (proxy feature — ON)", KNOWN_HOBBIES, index=KNOWN_HOBBIES.index("reading"))
        insured_occupation = c5.selectbox("Insured occupation (proxy feature — ON)", KNOWN_OCCUPATIONS, index=KNOWN_OCCUPATIONS.index("craft-repair"))
    else:
        insured_hobbies = insured_occupation = None
        st.caption("Hobby, occupation and ZIP code are not asked for: the shipped model does not use them "
                   "(lifestyle/occupation proxy features are switched off, and the ZIP feature was removed in PB-02).")

with tab_incident:
    c1, c2, c3 = st.columns(3)
    with c1:
        incident_severity = st.selectbox("Incident severity", ["Trivial Damage", "Minor Damage", "Major Damage", "Total Loss"],
                                         index=1, key="f_sev")
        incident_type = _cat("Incident type", "incident_type")
        collision_type = _cat("Collision type", "collision_type")
    with c2:
        incident_state = _cat("Incident state", "incident_state")
        authorities_contacted = _cat("Authorities contacted", "authorities_contacted")
        incident_hour = st.slider("Incident hour (0–23)", 0, 23, 12, key="f_hour")
    with c3:
        number_of_vehicles = st.number_input("Vehicles involved", 1, 4, 1, key="f_veh")
        bodily_injuries = st.number_input("Bodily injuries", 0, 2, 0, key="f_inj")
        witnesses = st.number_input("Witnesses", 0, 3, 1, key="f_wit")
    c4, c5 = st.columns(2)
    with c4:
        property_damage = _cat("Property damage?", "property_damage")
        police_report_available = _cat("Police report available?", "police_report_available")
    with c5:
        incident_date = st.date_input("Incident date", value=date.today(), min_value=date(1990, 1, 1),
                                      max_value=date.today(), key="incident_date")

with tab_amounts:
    c1, c2, c3, c4 = st.columns(4)
    injury_claim = c1.number_input("Injury claim ($)", 0.0, 100_000.0, 6800.0, step=100.0, key="f_ic")
    property_claim = c2.number_input("Property claim ($)", 0.0, 100_000.0, 13100.0, step=100.0, key="f_pc")
    vehicle_claim = c3.number_input("Vehicle claim ($)", 0.0, 150_000.0, 35100.0, step=100.0, key="f_vc")
    total_claim_amount = injury_claim + property_claim + vehicle_claim
    with c4:
        st.metric("Total claim (auto)", f"${total_claim_amount:,.0f}", help="Always injury + property + vehicle — "
                  "true for every one of the 1,000 training claims, so it is calculated, not typed.")
    auto_year = st.number_input("Vehicle model year", 1990, 2026, 2005, key="f_year")

# ---- Coverage check: what share of the model's weight this form supplies ----
from app.ml.form_spec import FORM_FIELDS  # single source of truth, checked by backend tests
field_imp = load_field_importance()
if not field_imp.empty:
    covered = field_imp["source_field"].map(lambda f: ENGINEERED_INPUTS.get(f, {f}) <= FORM_FIELDS)
    st.caption(f"This form supplies inputs for **{field_imp.loc[covered, 'share_of_total'].sum():.0%}** of the "
               f"model's SHAP weight (computed from `shap_importance_by_field.csv`, not typed).")

# ---- A5: validation (clear messages; scoring disabled until fixed) ----
errors = []
if incident_date < policy_bind_date:
    errors.append(f"Incident date ({incident_date}) is before the policy bind date ({policy_bind_date}).")
if auto_year > incident_date.year + 1:
    errors.append(f"Vehicle model year ({auto_year}) is after the incident year ({incident_date.year}).")
if min(injury_claim, property_claim, vehicle_claim) < 0 or policy_annual_premium <= 0:
    errors.append("Claim amounts cannot be negative and the annual premium must be positive.")
if not 16 <= age <= 100:
    errors.append("Insured age must be between 16 and 100.")
if months_as_customer / 12 > age:
    errors.append("Months as customer implies a relationship longer than the insured's age.")
for e in errors:
    st.error(e)

# ---- A5: derived values, read-only, recomputed live ----
_policy_age = (incident_date - policy_bind_date).days
st.markdown("##### Derived values (read-only)")
dv = st.columns(6)
dv[0].metric("Claim / premium", f"{total_claim_amount / policy_annual_premium:.1f}x" if policy_annual_premium else "—")
dv[1].metric("Injury share", f"{injury_claim / total_claim_amount:.0%}" if total_claim_amount else "—")
dv[2].metric("Property share", f"{property_claim / total_claim_amount:.0%}" if total_claim_amount else "—")
dv[3].metric("Vehicle share", f"{vehicle_claim / total_claim_amount:.0%}" if total_claim_amount else "—")
dv[4].metric("Policy age at incident", f"{_policy_age:,} days")
dv[5].metric("Vehicle age", f"{incident_date.year - auto_year} yrs")
st.caption(f"Incident day: {incident_date:%A} ({'weekend' if incident_date.weekday() >= 5 else 'weekday'}) — "
           "informational only, not a model feature.")

st.write("")
if st.button("Score claim", type="primary", width="stretch", disabled=bool(errors)):
    payload = {
        "months_as_customer": int(months_as_customer), "age": int(age), "policy_state": policy_state,
        "policy_csl": policy_csl, "policy_deductable": int(policy_deductable),
        "policy_annual_premium": float(policy_annual_premium), "umbrella_limit": int(umbrella_limit),
        "insured_sex": insured_sex, "insured_education_level": insured_education_level,
        "insured_relationship": insured_relationship, "capital-gains": int(capital_gains),
        "capital-loss": int(capital_loss), "incident_type": incident_type, "collision_type": collision_type,
        "incident_severity": incident_severity, "authorities_contacted": authorities_contacted,
        "incident_state": incident_state, "incident_hour_of_the_day": int(incident_hour),
        "number_of_vehicles_involved": int(number_of_vehicles), "property_damage": property_damage,
        "bodily_injuries": int(bodily_injuries), "witnesses": int(witnesses),
        "police_report_available": police_report_available,
        "injury_claim": float(injury_claim), "property_claim": float(property_claim),
        "vehicle_claim": float(vehicle_claim), "total_claim_amount": float(total_claim_amount),
        "auto_year": int(auto_year),
        "incident_date": incident_date.isoformat(), "policy_bind_date": policy_bind_date.isoformat(),
    }
    if INCLUDE_PROXY_FEATURES:
        payload["insured_hobbies"], payload["insured_occupation"] = insured_hobbies, insured_occupation

    result = service.score_one(payload)
    st.session_state["session_scored_count"] = st.session_state.get("session_scored_count", 0) + 1
    from app.db.persistence import persist_scored_claim
    db = new_db_session()
    try:
        claim = persist_scored_claim(db, payload, result, ingested_via="dashboard")
        db.commit()
        claim_id = claim.id
    finally:
        db.close()
    st.session_state["last_score"] = {"result": result, "payload": payload, "claim_id": claim_id}

if "last_score" in st.session_state:
    # UI-14: no gauge, no test-set histogram, no amount donut (the derived-values
    # strip already shows the shares). The score is a 0–1 fraud-risk score,
    # never a percentage. Band edges come from the model's risk-policy artifact.
    res = st.session_state["last_score"]["result"]
    p, thr, grade = res["fraud_probability"], res["operating_threshold"], res["risk_grade"]
    colour = {"Low": SUCCESS, "Medium": WARNING, "High": DANGER}[grade]
    edges = service.band_edges or {"medium_edge": MEDIUM_RISK_EDGE, "high_edge": HIGH_RISK_EDGE}

    st.divider()
    st.markdown(f"### {risk_badge(grade)} &nbsp; Fraud-risk score {p:.2f}", unsafe_allow_html=True)
    oof_path = PROCESSED_DIR / "dev_oof_scores.csv"
    pct_text = ""
    if oof_path.exists():
        ref = pd.read_csv(oof_path, usecols=["oof_score"])["oof_score"].to_numpy()
        pct_text = (f"<br/>Scores higher than {(ref < p).mean():.0%} of the {len(ref)} development claims "
                    "(out-of-fold scores, so no claim is scored by a model that saw it).")
    st.markdown(
        f"<div class='aeg-card' style='border-color:{colour}88'>"
        f"<b>{grade} band</b> · <b>{'Flagged for review' if p >= thr else 'Not flagged'}</b>"
        f"<br/><b>Recommended next step:</b> {res['recommended_action']}{pct_text}"
        f"<br/><span style='color:{MUTED};font-size:12px'>Bands: Low &lt; {edges['medium_edge']:.2f} ≤ Medium &lt; "
        f"{edges['high_edge']:.2f} ≤ High · review threshold {thr:.2f} · saved as claim "
        f"#{st.session_state['last_score']['claim_id']} · model {res['model_version']}</span></div>",
        unsafe_allow_html=True)
    st.write("")
    f1c, f2c = st.columns(2)
    f1c.markdown("**Factors increasing risk**\n" + ("\n".join(f"- {r['sentence']}" for r in res["factors_increasing_risk"][:3]) or "- none"))
    f2c.markdown("**Factors reducing risk**\n" + ("\n".join(f"- {r['sentence']}" for r in res["factors_reducing_risk"][:3]) or "- none"))

    st.plotly_chart(reasons_bar(res["top_reasons"]), width="stretch")
    st.caption("Each bar is one claim field (categorical answers are grouped, so a field appears once, with the "
               "value you entered). Red ▲ raises the risk score, green ▼ lowers it. Bar length = SHAP contribution.")
    with st.expander("All reason codes (table)"):
        st.dataframe(pd.DataFrame(res["top_reasons"])[["rank", "display_name", "display_value", "direction", "impact", "shap_value"]],
                     width="stretch", hide_index=True)
