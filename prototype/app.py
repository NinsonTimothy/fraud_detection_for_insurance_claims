"""
Explainable Risk Scoring — Insurance Claim Fraud Detection
Streamlit prototype: single-claim risk scoring with SHAP explanation.

University of Ghana FYP — Kwabena Adipah Osei & Timothy Ninson
Supervisor: Prof. Ebenezer Owusu

This file is UI ONLY. All scoring / feature-engineering logic lives in scoring.py so it
can be unit-tested (tests/test_prototype.py) and reused in 06_prototype.ipynb without
duplication. Feature formulas in scoring.py are confirmed against 02_preprocessing.ipynb;
two open gaps (zip3 lookup table, policy_state values) are documented there.

HOW TO RUN
----------
1. Place this file and scoring.py inside your project so that ../models/ and
   ../data/processed/ resolve correctly (i.e. run it from a `prototype/` folder sitting
   next to `models/` and `data/`), OR edit the paths at the top of scoring.py.
2. pip install streamlit shap joblib pandas numpy matplotlib scikit-learn xgboost imbalanced-learn
3. streamlit run app.py
"""

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from scoring import load_artifacts, build_feature_row, score_claim, SHAP_INTERPRETATIONS

US_STATES_IN_DATA = ["OH", "IN", "IL", "PA", "NY", "SC", "WV", "VA", "NC", "OR"]  # incident_state options
POLICY_STATES = ["OH", "IN", "IL"]  # inferred, standard for this dataset — confirm against your EDA notebook
INCIDENT_TYPES = ["Single Vehicle Collision", "Multi-vehicle Collision", "Vehicle Theft", "Parked Car"]
COLLISION_TYPES = ["Front Collision", "Rear Collision", "Side Collision", "Not Applicable"]
SEVERITIES = ["Trivial Damage", "Minor Damage", "Major Damage", "Total Loss"]
AUTHORITIES = ["Police", "Fire", "Ambulance", "Other", "None"]
HOBBIES = ["chess", "cross-fit", "reading", "camping", "kayaking", "golf", "hiking",
           "video-games", "sleeping", "yachting", "board-games", "bungie-jumping",
           "movies", "basketball", "polo", "skydiving", "paintball", "dancing",
           "base-jumping", "exercise"]  # TODO-VERIFY full list against your dataset


# --------------------------------------------------------------------------------------
# Cached artifact loading (Streamlit-specific — stays in this file, not scoring.py)
# --------------------------------------------------------------------------------------
@st.cache_resource
def get_artifacts():
    try:
        model, explainer, template_columns = load_artifacts()
        return model, explainer, template_columns, []
    except FileNotFoundError as e:
        return None, None, None, str(e).split("\n")[1:]


# --------------------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="Fraud Risk Scoring Prototype", layout="wide")
st.title("Explainable Risk Scoring — Insurance Claim Fraud Detection")
st.caption(
    "University of Ghana FYP prototype · Random Forest + SHAP · "
    "Kwabena Adipah Osei & Timothy Ninson · Supervisor: Prof. Ebenezer Owusu"
)

model, explainer, template_columns, missing_files = get_artifacts()

if missing_files:
    st.error(
        "Could not find required model artifacts. This app must be run from a folder "
        "next to your `models/` and `data/processed/` directories.\n\n"
        "Missing:\n" + "\n".join(missing_files)
    )
    st.stop()

st.warning(
    "⚠️ Two known gaps remain. ZIP3 risk tier can't be scored yet because the fraud-rate "
    "lookup table isn't saved anywhere in 02_preprocessing.ipynb (see scoring.py for the "
    "one-line fix needed there). Policy state options (OH/IN/IL) are inferred from the "
    "standard version of this dataset, not confirmed against your own EDA notebook. "
    "Everything else has been checked against 02_preprocessing.ipynb directly.",
    icon="⚠️",
)

with st.form("claim_form"):
    st.subheader("Claim & Policy Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Policyholder age", 18, 100, 35)
        months_as_customer = st.number_input("Months as customer", 0, 600, 120)
        policy_annual_premium = st.number_input("Policy annual premium ($)", 0.0, 5000.0, 1250.0)
        policy_deductable = st.selectbox("Policy deductible ($)", [500, 1000, 2000])
        umbrella_limit = st.number_input("Umbrella limit ($)", 0, 10_000_000, 0, step=1_000_000)
        capital_gains = st.number_input("Capital gains ($)", 0, 100000, 0)
        capital_loss = st.number_input("Capital loss ($)", -100000, 0, 0)
        csl_per_person = st.number_input("CSL per person ($)", 0, 500000, 100000)
        csl_total = st.number_input("CSL total ($)", 0, 1000000, 300000)

    with col2:
        incident_type = st.selectbox("Incident type", INCIDENT_TYPES)
        collision_type = st.selectbox("Collision type", COLLISION_TYPES)
        incident_severity = st.selectbox("Incident severity", SEVERITIES)
        authorities_contacted = st.selectbox("Authorities contacted", AUTHORITIES)
        incident_state = st.selectbox("Incident state", US_STATES_IN_DATA)
        incident_hour_of_the_day = st.slider("Incident hour (0-23)", 0, 23, 14)
        number_of_vehicles_involved = st.number_input("Number of vehicles involved", 1, 6, 1)
        bodily_injuries = st.number_input("Bodily injuries", 0, 5, 0)
        witnesses = st.number_input("Witnesses", 0, 10, 1)

    with col3:
        injury_claim = st.number_input("Injury claim ($)", 0, 200000, 5000)
        property_claim = st.number_input("Property claim ($)", 0, 200000, 5000)
        vehicle_claim = st.number_input("Vehicle claim ($)", 0, 200000, 10000)
        insured_hobbies = st.selectbox("Insured hobbies", HOBBIES)
        insured_occupation = st.text_input("Insured occupation (e.g. exec-managerial)", "craft-repair")
        auto_region = st.selectbox("Auto region", ["USA", "Germany", "Japan", "Other"])
        auto_year = st.number_input("Vehicle model year", 1990, 2026, 2015)
        policy_state = st.selectbox("Policy state (where the policy was issued)", POLICY_STATES)

    st.subheader("Dates (for policy-age calculation)")
    d1, d2 = st.columns(2)
    with d1:
        policy_bind_date = st.date_input("Policy bind date")
    with d2:
        incident_date = st.date_input("Incident date")

    submitted = st.form_submit_button("Score this claim", type="primary")

if submitted:
    raw = dict(
        age=age, months_as_customer=months_as_customer, policy_annual_premium=policy_annual_premium,
        policy_deductable=policy_deductable, umbrella_limit=umbrella_limit, capital_gains=capital_gains,
        capital_loss=capital_loss, csl_per_person=csl_per_person, csl_total=csl_total,
        incident_type=incident_type, collision_type=collision_type, incident_severity=incident_severity,
        authorities_contacted=authorities_contacted, incident_state=incident_state,
        incident_hour_of_the_day=incident_hour_of_the_day,
        number_of_vehicles_involved=number_of_vehicles_involved, bodily_injuries=bodily_injuries,
        witnesses=witnesses, injury_claim=injury_claim, property_claim=property_claim,
        vehicle_claim=vehicle_claim, insured_hobbies=insured_hobbies,
        insured_occupation=insured_occupation, auto_region=auto_region, auto_year=auto_year,
        policy_state=policy_state,
        policy_bind_date=pd.Timestamp(policy_bind_date), incident_date=pd.Timestamp(incident_date),
    )

    feature_row, unmatched = build_feature_row(raw, template_columns)
    result = score_claim(model, explainer, feature_row, top_n=8)

    risk_band = result["risk_band"]
    band_color = {"Low": "green", "Medium": "orange", "High": "red"}[risk_band]

    st.divider()
    st.subheader("Result")

    r1, r2, r3 = st.columns(3)
    r1.metric("Fraud probability", f"{result['fraud_probability']:.1%}")
    r2.metric("Risk score (0-100)", result["risk_score"])
    r3.markdown(f"<h3 style='color:{band_color};'>Risk band: {risk_band}</h3>", unsafe_allow_html=True)

    if risk_band == "High":
        st.error("Recommended action: refer to fraud investigation team.")
    elif risk_band == "Medium":
        st.warning("Recommended action: request additional documentation.")
    else:
        st.success("Recommended action: process normally.")

    if unmatched:
        st.info(
            "Some selected categories didn't match a known training column (they may be the "
            "reference/dropped category, or genuinely unseen). This is expected for one "
            "category per categorical field:\n\n" + "\n".join(f"- {u}" for u in unmatched)
        )

    st.subheader("Why this score — SHAP explanation")
    top_factors = result["top_shap_factors"]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["crimson" if v > 0 else "steelblue" for v in top_factors.values[::-1]]
    ax.barh(top_factors.index[::-1], top_factors.values[::-1], color=colors, edgecolor="white")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("SHAP value (impact on fraud probability)")
    ax.set_title("Top factors for this claim (red = pushes toward fraud, blue = pushes away)")
    st.pyplot(fig)

    st.markdown("**Plain-language breakdown:**")
    for feat, val in top_factors.items():
        direction = "increases" if val > 0 else "decreases"
        note = SHAP_INTERPRETATIONS.get(feat, "")
        actual_val = feature_row.at[0, feat]
        st.markdown(f"- **{feat}** (value: `{actual_val}`) {direction} fraud probability. {note}")

    with st.expander("Show raw 94-column feature vector sent to the model"):
        st.dataframe(feature_row.T.rename(columns={0: "value"}))
