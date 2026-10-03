"""
scoring.py — Shared, pure backend logic for the Explainable Risk Scoring prototype.

This module has NO Streamlit dependency and does NO I/O side effects beyond loading
model artifacts. That's the point: `app.py` (the UI) and `tests/test_prototype.py`
(the validation suite) both import from here, so there is exactly one implementation
of the scoring logic instead of two copies that can silently drift apart.

============================== IMPORTANT — READ BEFORE TRUSTING OUTPUT ==============================
Feature formulas below are confirmed against 02_preprocessing.ipynb. Two gaps remain unresolved:

1. zip3_risk_tier has no lookup table saved anywhere in the notebook, so it can't be computed for
   a new claim yet. See the comment in build_feature_row for the one-line fix needed in the notebook.
2. policy_state was added as a one-hot field based on this being the standard OH/IN/IL insurance
   fraud dataset. That's an inference, not something confirmed line-by-line in the notebook, worth
   a quick check against your own EDA output before relying on it for a demo.
=======================================================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Paths — adjust if your folder layout differs. Relative to wherever the importing
# script/notebook is run from (both app.py and the notebook expect ../models, ../data).
# --------------------------------------------------------------------------------------
MODEL_DIR = "../models"
DATA_DIR = "../data/processed"

RF_MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_final.pkl")
SHAP_EXPLAINER_PATH = os.path.join(MODEL_DIR, "shap_explainer.pkl")
X_TRAIN_PATH = os.path.join(DATA_DIR, "X_train.csv")

# Decision threshold from 05_shap_explanations.ipynb (Phase 3 best-model evaluation)
RISK_THRESHOLD = 0.55

# Top-10 SHAP feature interpretations, lifted directly from your notebook (cell 11)
SHAP_INTERPRETATIONS = {
    "incident_severity": "Higher severity (Major Damage, Total Loss) strongly pushes predictions toward fraud.",
    "is_major_damage": "Claims flagged as Major Damage are significantly more likely to be predicted fraud.",
    "is_highrisk_hobby": "Chess and cross-fit hobbies are associated with higher fraud probability.",
    "zip3_risk_tier_low_risk": "Policyholders in high-risk ZIP3 areas receive higher fraud scores — the project's novel geographic feature.",
    "vehicle_claim": "Larger vehicle claim amounts increase fraud probability.",
    "injury_claim": "High injury claim amounts push predictions toward fraud.",
    "claim_to_premium_ratio": "Claims disproportionately large relative to the premium paid are flagged as suspicious.",
    "is_exec_occupation": "Executive/managerial occupations are associated with slightly higher fraud probability.",
    "witnesses": "Fewer witnesses increases fraud probability — consistent with staged accident patterns.",
    "is_no_witness": "Zero witnesses is a direct fraud signal — no independent party can contradict the claim.",
    "authorities_contacted_Police": "Whether police were the contacted authority shifts fraud probability.",
    "incident_state_WV": "Geographic clustering of fraud by incident state.",
    "incident_state_NY": "Geographic clustering of fraud by incident state.",
    "number_of_vehicles_involved": "More vehicles involved changes the fraud probability profile.",
    "collision_type_Not Applicable": "Collision type recorded as Not Applicable shifts fraud probability.",
    "hour_bucket_Night": "Incidents reported at night are associated with higher fraud probability.",
}

HIGH_RISK_HOBBIES = {"chess", "cross-fit"}  # confirmed against 02_preprocessing.ipynb
EXEC_OCCUPATIONS = {"exec-managerial"}  # confirmed against 02_preprocessing.ipynb

# ZIP3 risk tiers, confirmed against 02_preprocessing.ipynb. Computed as a per-ZIP3
# average fraud rate, then binned into low_risk / baseline / elevated / high_risk.
# One-hot encoded with drop_first=True, and since pd.get_dummies sorts categories
# alphabetically, "baseline" is the dropped reference category.
# STILL UNRESOLVED: the fraud-rate-by-ZIP3 lookup table itself is never saved to
# disk in the notebook, so this prototype has no way to compute the tier for a new
# claim. See the zip3_risk_tier block in build_feature_row for what's needed.


# --------------------------------------------------------------------------------------
# Artifact loading
# --------------------------------------------------------------------------------------
def artifacts_available() -> list:
    """Returns a list of any missing required files. Empty list means everything's present."""
    return [p for p in [RF_MODEL_PATH, SHAP_EXPLAINER_PATH, X_TRAIN_PATH] if not os.path.exists(p)]


def load_artifacts():
    """
    Loads the trained RF pipeline, the fitted SHAP TreeExplainer, and the training
    column schema (used only to align feature vectors, not for row values).
    Raises FileNotFoundError with a clear message if anything is missing.
    """
    missing = artifacts_available()
    if missing:
        raise FileNotFoundError(
            "Missing required model artifacts:\n" + "\n".join(f"  - {m}" for m in missing)
        )
    model = joblib.load(RF_MODEL_PATH)
    explainer = joblib.load(SHAP_EXPLAINER_PATH)
    x_train_template = pd.read_csv(X_TRAIN_PATH, nrows=5)
    x_train_template = x_train_template.rename(
        columns={"capital-gains": "capital_gains", "capital-loss": "capital_loss"}
    )
    return model, explainer, x_train_template.columns


# --------------------------------------------------------------------------------------
# Risk score / band conversion — confirmed from 05_shap_explanations.ipynb
# --------------------------------------------------------------------------------------
def assign_band(score: int) -> str:
    if score < 30:
        return "Low"
    elif score < 60:
        return "Medium"
    return "High"


def fraud_risk_score(probability: float) -> dict:
    score = int(round(probability * 100))
    return {"probability": round(float(probability), 4), "risk_score": score, "risk_band": assign_band(score)}


# --------------------------------------------------------------------------------------
# Feature engineering — build the 94-column vector the model expects
# --------------------------------------------------------------------------------------
def build_feature_row(raw: dict, template_columns: pd.Index):
    """
    Takes raw form/claim inputs and returns (row_df, unmatched_categories), where
    row_df is a single-row DataFrame aligned to template_columns.

    See module docstring — several fields below are marked TODO-VERIFY.
    """
    row = pd.DataFrame(np.zeros((1, len(template_columns))), columns=template_columns)
    unmatched_categories = []

    # ---- confirmed raw numeric fields (STANDARD_COLS / MINMAX_COLS / ROBUST_COLS in 03_modelling.ipynb) ----
    direct_numeric = {
        "months_as_customer": raw["months_as_customer"],
        "age": raw["age"],
        "policy_annual_premium": raw["policy_annual_premium"],
        "policy_deductable": raw["policy_deductable"],
        "incident_hour_of_the_day": raw["incident_hour_of_the_day"],
        "number_of_vehicles_involved": raw["number_of_vehicles_involved"],
        "bodily_injuries": raw["bodily_injuries"],
        "witnesses": raw["witnesses"],
        "capital_gains": raw["capital_gains"],
        "capital_loss": raw["capital_loss"],
        "injury_claim": raw["injury_claim"],
        "property_claim": raw["property_claim"],
        "vehicle_claim": raw["vehicle_claim"],
        "umbrella_limit": raw["umbrella_limit"],
    }
    for col, val in direct_numeric.items():
        if col in row.columns:
            row.at[0, col] = val

    # ---- engineered numeric features, confirmed against 02_preprocessing.ipynb ----
    total_claim = raw["injury_claim"] + raw["property_claim"] + raw["vehicle_claim"]
    claim_to_premium_ratio = (total_claim / raw["policy_annual_premium"]) if raw["policy_annual_premium"] else 0
    # notebook divides by (total_claim_amount + 1), not a plain conditional zero-check
    vehicle_claim_pct = raw["vehicle_claim"] / (total_claim + 1)
    injury_claim_pct = raw["injury_claim"] / (total_claim + 1)
    # notebook computes car_age from incident_date.dt.year, not a separately-entered year
    car_age = raw["incident_date"].year - raw["auto_year"]
    policy_age_at_incident_days = (raw["incident_date"] - raw["policy_bind_date"]).days
    is_new_customer = 1 if policy_age_at_incident_days < 180 else 0
    is_weekend = 1 if raw["incident_date"].dayofweek in (5, 6) else 0

    engineered_numeric = {
        "claim_to_premium_ratio": claim_to_premium_ratio,
        "vehicle_claim_pct": vehicle_claim_pct,
        "injury_claim_pct": injury_claim_pct,
        "car_age": car_age,
        "policy_age_at_incident_days": policy_age_at_incident_days,
        "csl_per_person": raw["csl_per_person"],
        "csl_total": raw["csl_total"],
    }
    for col, val in engineered_numeric.items():
        if col in row.columns:
            row.at[0, col] = val

    # ---- engineered binary flags (TODO-VERIFY thresholds/definitions) ----
    is_major_damage = 1 if raw["incident_severity"] == "Major Damage" else 0
    is_no_witness = 1 if raw["witnesses"] == 0 else 0
    is_highrisk_hobby = 1 if raw["insured_hobbies"] in HIGH_RISK_HOBBIES else 0
    is_exec_occupation = 1 if raw["insured_occupation"] in EXEC_OCCUPATIONS else 0

    for col, val in {
        "is_major_damage": is_major_damage,
        "is_no_witness": is_no_witness,
        "is_highrisk_hobby": is_highrisk_hobby,
        "is_exec_occupation": is_exec_occupation,
        "is_new_customer": is_new_customer,
        "is_weekend": is_weekend,
    }.items():
        if col in row.columns:
            row.at[0, col] = val

    # ---- incident_severity ordinal encoding (TODO-VERIFY exact mapping) ----
    severity_ordinal_map = {"Trivial Damage": 0, "Minor Damage": 1, "Major Damage": 2, "Total Loss": 3}
    if "incident_severity" in row.columns:
        row.at[0, "incident_severity"] = severity_ordinal_map.get(raw["incident_severity"], 0)

    # ---- hour_bucket, confirmed against 02_preprocessing.ipynb ----
    # Real buckets: Night (22:00-05:59), Rush_Hour (07:00-09:59 and 17:00-19:59),
    # Daytime (everything else). Daytime is the dropped reference category since
    # pd.get_dummies(drop_first=True) drops it first alphabetically.
    hour = raw["incident_hour_of_the_day"]
    if hour >= 22 or hour <= 5:
        hour_bucket = "Night"
    elif (7 <= hour <= 9) or (17 <= hour <= 19):
        hour_bucket = "Rush_Hour"
    else:
        hour_bucket = "Daytime"
    hour_col = f"hour_bucket_{hour_bucket}"
    if hour_col in row.columns:
        row.at[0, hour_col] = 1
    elif hour_bucket != "Daytime":  # Daytime is the dropped reference category
        unmatched_categories.append(hour_col)

    # ---- zip3_risk_tier — STILL UNRESOLVED ----
    # 02_preprocessing.ipynb confirms this is a real 4-tier feature (low_risk,
    # baseline, elevated, high_risk) derived from a per-ZIP3 average fraud rate,
    # one-hot encoded with "baseline" as the dropped reference. But the ZIP3 ->
    # fraud-rate lookup table is computed and then dropped in that notebook, it's
    # never written to disk. Without it, there's no way to score a new claim's
    # ZIP3 correctly here. To close this: add a line to 02_preprocessing.ipynb
    # saving `zip3_fraud` to ../data/processed/zip3_fraud_lookup.csv, then this
    # app can look up the tier from an insured ZIP code the same way training did.
    # Until then, all zip3_risk_tier_* dummy columns are left at 0, which silently
    # scores every claim as if its ZIP3 fell in the dropped "baseline" tier.
    for tier_col in ("zip3_risk_tier_low_risk", "zip3_risk_tier_elevated", "zip3_risk_tier_high_risk"):
        if tier_col in row.columns:
            row.at[0, tier_col] = 0

    # ---- one-hot categorical fields ----
    one_hot_selections = {
        "incident_type": raw["incident_type"],
        "collision_type": raw["collision_type"],
        "authorities_contacted": raw["authorities_contacted"],
        "incident_state": raw["incident_state"],
        "insured_hobbies": raw["insured_hobbies"],
        "auto_region": raw["auto_region"],
        "policy_state": raw["policy_state"],
    }
    for prefix, selection in one_hot_selections.items():
        col_name = f"{prefix}_{selection}"
        if col_name in row.columns:
            row.at[0, col_name] = 1
        else:
            unmatched_categories.append(col_name)

    return row, unmatched_categories


# --------------------------------------------------------------------------------------
# End-to-end scoring — what app.py calls when the form is submitted
# --------------------------------------------------------------------------------------
def score_claim(model, explainer, feature_row: pd.DataFrame, top_n: int = 8) -> dict:
    """
    Runs the RF pipeline + SHAP explainer on a single-row feature DataFrame.
    RF pipeline expects raw (unscaled) columns — confirmed from 03_modelling.ipynb,
    since Random Forest was evaluated on X_test, not X_test_scaled.
    """
    fraud_prob = model.predict_proba(feature_row)[0, 1]
    risk = fraud_risk_score(fraud_prob)

    shap_values = explainer.shap_values(feature_row)
    if isinstance(shap_values, list):
        shap_row = shap_values[1][0]
    else:
        shap_row = shap_values[0, :, 1]

    shap_series = pd.Series(shap_row, index=feature_row.columns).sort_values(key=abs, ascending=False)

    return {
        "fraud_probability": fraud_prob,
        "risk_score": risk["risk_score"],
        "risk_band": risk["risk_band"],
        "top_shap_factors": shap_series.head(top_n),
    }
