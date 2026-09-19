"""
feature_engineering.py — turns the 37 raw claim columns (after clean_data.py)
into the model's actual feature matrix, and is the ONE place both training
and scoring (live API, batch upload, Oracle external-validation adapter) go
through, so they can never silently drift apart.

Raw schema this expects (35 columns `RAW_FEATURE_COLUMNS`, i.e. everything
except the label `fraud_reported`) is documented at the bottom of this file.

Three engineered feature groups are worth reading before touching this file,
because they are exactly the three the project's own feature critique
(`docs/ml_feature_critique.md`) flags as NOT safe to treat as genuine fraud
signal — kept in because dropping them silently would misrepresent what the
shipped model in this build actually does, but each is deliberately,
individually inspectable via `RISKY_FEATURE_COLUMNS` below:

  1. `is_highrisk_hobby` / `insured_hobbies_chess` / `insured_hobbies_cross-fit`
     — chess (82.6% fraud, n=46) and cross-fit (74.3%, n=35) claimants are
     fraud-flagged far more than every other hobby (17-30%) in this 1,000-row
     dataset. There is no causal fraud mechanism here; this is almost
     certainly a small-sample/synthetic-generation artifact, and leaning on
     it is close to proxy-discrimination (lifestyle -> risk score) real
     insurance regulators scrutinize. Kept, but flagged.
  2. `zip3_risk_tier_*` — target-encoded from the SAME 1,000 rows the model
     trains on (not a held-out fold), a leakage pattern the original
     project's own preprocessing notebook already self-documented but never
     fixed. It is real, measurable leakage: this feature's information
     content partially comes from having seen the label.
  3. `is_exec_occupation` — same shape as (1) at lower severity (36.8% vs.
     24.7% base rate, n=76).

For the honest fix, see `docs/LIMITATIONS.md` — recomputing zip3 per-CV-fold
or dropping (1)/(3) entirely are both one-line changes in `train.py`, left
as an explicit, disclosed design choice rather than silently done for you.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# Raw schema — every column a scoring request (live API / batch upload /
# an external adapter like oracle_adapter.py) may supply. Anything absent
# falls back to MISSING_COLUMN_DEFAULTS below, exactly as documented in
# `docs/generalization_and_cv_results.md`'s Oracle adapter section.
# ---------------------------------------------------------------------------
RAW_FEATURE_COLUMNS = [
    "months_as_customer", "age", "policy_bind_date", "policy_state", "policy_csl",
    "policy_deductable", "policy_annual_premium", "umbrella_limit", "insured_zip",
    "insured_sex", "insured_education_level", "insured_occupation", "insured_hobbies",
    "insured_relationship", "capital-gains", "capital-loss", "incident_date",
    "incident_type", "collision_type", "incident_severity", "authorities_contacted",
    "incident_state", "incident_hour_of_the_day", "number_of_vehicles_involved",
    "property_damage", "bodily_injuries", "witnesses", "police_report_available",
    "total_claim_amount", "injury_claim", "property_claim", "vehicle_claim",
    "auto_make", "auto_year",
]

MISSING_COLUMN_DEFAULTS: dict[str, object] = {
    "months_as_customer": 150, "age": 39, "policy_bind_date": "2005-01-01",
    "policy_state": "OH", "policy_csl": "250/500", "policy_deductable": 1000,
    "policy_annual_premium": 1256.0, "umbrella_limit": 0, "insured_zip": 500000,
    "insured_sex": "MALE", "insured_education_level": "High School",
    "insured_occupation": "other-service", "insured_hobbies": "reading",
    "insured_relationship": "not-in-family", "capital-gains": 0, "capital-loss": 0,
    "incident_date": "2015-01-15", "incident_type": "Multi-vehicle Collision",
    "collision_type": "Rear Collision", "incident_severity": "Minor Damage",
    "authorities_contacted": "Police", "incident_state": "OH",
    "incident_hour_of_the_day": 12, "number_of_vehicles_involved": 1,
    "property_damage": "NO", "bodily_injuries": 0, "witnesses": 1,
    "police_report_available": "NO", "total_claim_amount": 52800.0,
    "injury_claim": 6800.0, "property_claim": 13100.0, "vehicle_claim": 35100.0,
    "auto_make": "Ford", "auto_year": 2005,
}

RISKY_FEATURE_COLUMNS = [
    "is_highrisk_hobby", "is_exec_occupation", "zip3_risk_tier_low_risk",
    "zip3_risk_tier_medium_risk", "zip3_risk_tier_high_risk",
]

HIGH_RISK_HOBBIES = {"chess", "cross-fit"}
SEVERITY_ORDINAL = {"Trivial Damage": 0, "Minor Damage": 1, "Major Damage": 2, "Total Loss": 3}

# Columns one-hot-encoded outright (low-to-moderate cardinality).
# NOTE: `insured_hobbies` (20 levels), `insured_occupation` (14), and
# `auto_make` (14) are DELIBERATELY excluded from full one-hot encoding —
# an early version of this rebuild included them and cross-validated at
# ROC-AUC 0.94 but collapsed to 0.59-0.78 on a genuine single holdout split
# (checked across 6 random seeds), the signature of a high-cardinality/
# low-row-count overfitting problem, not a real generalizable signal. This
# is exactly what `docs/ml_feature_critique.md` already recommends for
# `insured_hobbies`/`insured_occupation` ("drop, or replace with something
# causally grounded") — implemented here rather than left as an unapplied
# recommendation. Their one documented, genuinely-informative signal
# (`is_highrisk_hobby`, `is_exec_occupation`) is kept as a single flag each
# instead of a full one-hot block. `auto_make` has no documented predictive
# value anywhere in the critique and was never flagged as useful, so it's
# dropped outright rather than kept for the sake of using every raw column.
CATEGORICAL_COLUMNS = [
    "policy_state", "policy_csl", "insured_sex", "insured_education_level",
    "insured_relationship", "incident_type", "collision_type",
    "authorities_contacted", "incident_state", "property_damage",
    "police_report_available",
]

NUMERIC_PASSTHROUGH_COLUMNS = [
    "months_as_customer", "age", "policy_deductable", "policy_annual_premium",
    "umbrella_limit", "capital-gains", "capital-loss", "incident_hour_of_the_day",
    "number_of_vehicles_involved", "bodily_injuries", "witnesses",
    "total_claim_amount", "injury_claim", "property_claim", "vehicle_claim", "auto_year",
]


def apply_missing_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """Fill any raw column an adapter didn't supply with its documented
    fallback default. This is what makes `oracle_adapter.py` honest: fields
    Oracle genuinely has get real values, everything else gets a disclosed,
    neutral constant — nothing invented to move the score either way."""
    df = df.copy()
    for col in RAW_FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = MISSING_COLUMN_DEFAULTS[col]
        else:
            df[col] = df[col].fillna(MISSING_COLUMN_DEFAULTS[col])
    return df


def _zip3_lookup_from_training(df: pd.DataFrame, fraud_col: pd.Series) -> pd.DataFrame:
    """Builds the target-encoded ZIP3->fraud-rate lookup table used by
    `zip3_risk_tier`. Documented, disclosed leakage (see module docstring
    point 2): built from the SAME rows it will be applied to at train time.
    Saved to disk so scoring time reuses the exact training-time table."""
    zip3 = (df["insured_zip"].astype(int) // 100).rename("zip3")
    tmp = pd.DataFrame({"zip3": zip3, "fraud": fraud_col.values})
    lookup = tmp.groupby("zip3")["fraud"].mean().rename("zip3_fraud_rate").reset_index()
    return lookup


def engineer_features(
    df: pd.DataFrame,
    zip3_lookup: pd.DataFrame,
    fit_mode: bool = False,
) -> pd.DataFrame:
    """The single feature-engineering pipeline used by training, the live
    API, batch scoring, and the Oracle adapter. `zip3_lookup` must be the
    table saved at training time (see `_zip3_lookup_from_training`) — never
    rebuilt at scoring time, or every scoring call would silently leak
    whatever batch it's currently scoring into its own risk tiers."""
    df = apply_missing_defaults(df)
    out = pd.DataFrame(index=df.index)

    # --- Numeric ratios (real signal — claim size vs. premium is a
    # standard actuarial red flag) ---
    premium = df["policy_annual_premium"].replace(0, np.nan)
    total_claim = df["total_claim_amount"].replace(0, np.nan)
    out["claim_to_premium_ratio"] = (df["total_claim_amount"] / premium).fillna(0.0)
    out["vehicle_claim_pct"] = (df["vehicle_claim"] / total_claim).fillna(0.0)
    out["injury_claim_pct"] = (df["injury_claim"] / total_claim).fillna(0.0)
    out["property_claim_pct"] = (df["property_claim"] / total_claim).fillna(0.0)

    # --- Date-derived (real signal — "new policy, big claim" is
    # well-documented) ---
    bind_date = pd.to_datetime(df["policy_bind_date"], errors="coerce")
    incident_date = pd.to_datetime(df["incident_date"], errors="coerce")
    policy_age_days = (incident_date - bind_date).dt.days
    out["policy_age_at_incident_days"] = policy_age_days.fillna(policy_age_days.median() if policy_age_days.notna().any() else 365)
    out["is_new_customer"] = (out["policy_age_at_incident_days"] < 30).astype(int)
    out["vehicle_age_at_incident"] = (incident_date.dt.year.fillna(incident_date.dt.year.median() if incident_date.notna().any() else 2015) - df["auto_year"]).clip(lower=0)

    # --- Behavioral / structural flags (real signal) ---
    out["is_no_witness"] = (df["witnesses"] == 0).astype(int)
    out["incident_severity_ordinal"] = df["incident_severity"].map(SEVERITY_ORDINAL).fillna(1).astype(int)  # TODO-VERIFY: ordinal order inferred, not confirmed against original preprocessing notebook — see docs/ml_feature_critique.md
    out["is_major_damage"] = (df["incident_severity"] == "Major Damage").astype(int)

    # --- Flagged-risky features (module docstring points 1 & 3) ---
    out["is_highrisk_hobby"] = df["insured_hobbies"].isin(HIGH_RISK_HOBBIES).astype(int)
    out["is_exec_occupation"] = (df["insured_occupation"] == "exec-managerial").astype(int)

    # --- ZIP3 risk tier (module docstring point 2 — disclosed leakage) ---
    zip3 = (df["insured_zip"].astype(int) // 100)
    merged = zip3.rename("zip3").to_frame().merge(zip3_lookup, on="zip3", how="left")
    fallback_rate = zip3_lookup["zip3_fraud_rate"].mean() if len(zip3_lookup) else 0.247
    fraud_rate = merged["zip3_fraud_rate"].fillna(fallback_rate).values
    tier = np.select(
        [fraud_rate < 0.20, fraud_rate < 0.35],
        ["low_risk", "medium_risk"],
        default="high_risk",
    )
    out["zip3_risk_tier"] = tier

    # --- Numeric passthrough ---
    for col in NUMERIC_PASSTHROUGH_COLUMNS:
        out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # --- One-hot encode categoricals (incl. insured_hobbies, zip3_risk_tier) ---
    cat_df = df[CATEGORICAL_COLUMNS].astype(str).copy()
    cat_df["zip3_risk_tier"] = out.pop("zip3_risk_tier")
    dummies = pd.get_dummies(cat_df, prefix=cat_df.columns, prefix_sep="_")
    out = pd.concat([out, dummies], axis=1)

    return out


def align_to_training_columns(df: pd.DataFrame, training_columns: list[str]) -> pd.DataFrame:
    """Reindexes a scored batch's one-hot columns against the exact columns
    seen at training time, filling any column this batch never produced
    with 0 (correctly meaning "not this category"), never NaN.

    This directly fixes the bug found during Aegis Risk Engine's build:
    naive `pd.get_dummies` on a SINGLE-row scoring request only emits dummy
    columns for that one row's own category, and reindexing without an
    explicit fill_value silently produces NaN ("missing") instead of 0
    ("not this category") for every other category — corrupting every
    downstream score. Regression test: backend/tests/test_ml_core.py."""
    return df.reindex(columns=training_columns, fill_value=0)
