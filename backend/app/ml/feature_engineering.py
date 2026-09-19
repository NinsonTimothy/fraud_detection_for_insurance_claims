"""
feature_engineering.py — turns the 37 raw claim columns (after clean_data.py)
into the model's actual feature matrix, and is the ONE place both training
and scoring (live API, batch upload, Oracle external-validation adapter) go
through, so they can never silently drift apart.

Raw schema this expects (35 columns `RAW_FEATURE_COLUMNS`, i.e. everything
except the label `fraud_reported`) is documented at the bottom of this file.

Two engineered feature groups are worth reading before touching this file,
because they are exactly the ones the project's own feature critique
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
  2. `is_exec_occupation` — same shape as (1) at lower severity (36.8% vs.
     24.7% base rate, n=76).

PB-02 (fixed, was point 2 here): this file used to also build a
`zip3_risk_tier_*` target-encoded feature from `insured_zip // 100`. That
comment was written under the assumption the result was a genuine 3-digit
ZIP3 prefix, matching Version A's original approach. It is not: this
dataset's `insured_zip` values are already 6-digit US ZIPs (e.g. 605280),
so `// 100` only drops the last 2 digits, leaving a 4-digit prefix that is
almost unique per row (515 distinct groups from 1,000 rows, median group
size 2.0, 90.9% of groups with <=3 rows). Combined with being fit on the
SAME rows it scores, this let the model memorize labels through the lookup
table: 0.2% fraud in the "low risk" tier vs. 71.8% in "high risk" on
TRAIN, collapsing to a flat ~20-28% on TEST — and it consumed 53.2% of
total SHAP weight while contributing nothing that generalized (holdout
ROC-AUC 0.656 with it in, one root cause of this build measuring far below
Project A's 0.860). Disclosing this as "leakage, kept in" was itself
wrong once the actual bug (4-digit, not 3-digit, prefix) was understood —
there is no honest version of a near-row-unique lookup table, so it is
removed entirely rather than kept and disclosed. `insured_zip` remains in
`RAW_FEATURE_COLUMNS` / `MISSING_COLUMN_DEFAULTS` as an EDA-only raw field;
no engineered feature is derived from it anymore. See
`docs/REBUILD_NOTES.md` for the full before/after evidence.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.core.config import INCLUDE_PROXY_FEATURES

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
    "is_highrisk_hobby", "is_exec_occupation",
]

HIGH_RISK_HOBBIES = {"chess", "cross-fit"}
# PB-24: this rank order is CONFIRMED, not inferred — the original FYP
# project's own 02_preprocessing.ipynb / 03_modelling.ipynb encode
# incident_severity with sklearn.OrdinalEncoder using this exact table
# (Trivial Damage < Minor Damage < Major Damage < Total Loss), verified
# directly against that notebook's source during the Version A vs.
# Version B comparison (see the "Version A vs Version B comparison" doc).
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
#
# PB-24: insured_education_level is deliberately one-hot here, NOT ordinal.
# The original FYP project ordinal-encoded it with the order JD < High
# School < Associate < College < Masters < PhD < MD — ranking a doctoral
# law degree below a high-school diploma is not a real ordering, so that
# scheme wasn't carried over; one-hot makes no ordering claim at all.
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


def engineer_features(df: pd.DataFrame, include_proxy_features: bool | None = None) -> pd.DataFrame:
    """The single feature-engineering pipeline used by training, the live
    API, batch scoring, and the Oracle adapter.

    SH-02 / D3: `include_proxy_features` gates `RISKY_FEATURE_COLUMNS`
    (`is_highrisk_hobby`, `is_exec_occupation` — see module docstring for
    why they're flagged). Defaults to `None`, which reads
    `app.core.config.INCLUDE_PROXY_FEATURES` (itself defaulting to
    `False` — the deployable headline model excludes them). Pass an
    explicit `True`/`False` to build a specific variant regardless of
    config, e.g. to compute the with/without ablation comparison in
    `train.py` — both call sites need both variants from the same
    function, not just whatever the process-wide config says."""
    if include_proxy_features is None:
        include_proxy_features = INCLUDE_PROXY_FEATURES
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
    # SH-06: clip(lower=0) — one row in the cleaned dataset has
    # incident_date before policy_bind_date (a -20 day "age"), a data
    # quality artifact, not a real pre-bind claim.
    out["policy_age_at_incident_days"] = policy_age_days.fillna(policy_age_days.median() if policy_age_days.notna().any() else 365).clip(lower=0)

    # SH-06 (fixed): is_new_customer used to threshold
    # `policy_age_at_incident_days < 30`. Reproduced against the cleaned
    # dataset: that field's mean is ~4,739 days (~13 years) and its "new"
    # group is near-empty at EVERY day-threshold tested (0.2% of rows at
    # 30 days, only 1.7% even at 180 days) — incident_date/policy_bind_date
    # in this dataset don't actually encode short-tenure relationships, so
    # no choice of day-threshold on this field can produce a usable
    # feature. `months_as_customer` (a real, directly-supplied raw field,
    # not derived from two dates) shows the documented "new policy, big
    # claim" pattern properly: fraud rate rises from the 24.6% baseline to
    # 29.2%/34.4%/34.1% at <12/<18/<24-month cutoffs, with reasonably
    # sized groups (24/32/41 rows). A Fisher exact test across a 6-36
    # month grid found the strongest (though not conventionally
    # significant at this sample size — lowest p~=0.17 at 21 months)
    # separation around 18-24 months; 24 months (a standard "new business"
    # underwriting window) is used here. See docs/REBUILD_NOTES.md for the
    # full grid. This is a materially weak, low-confidence signal — kept
    # because it's directionally consistent with the literature and does
    # no harm, not because it's a strong predictor in this small dataset.
    out["is_new_customer"] = (df["months_as_customer"] < 24).astype(int)
    out["vehicle_age_at_incident"] = (incident_date.dt.year.fillna(incident_date.dt.year.median() if incident_date.notna().any() else 2015) - df["auto_year"]).clip(lower=0)

    # --- Behavioral / structural flags (real signal) ---
    out["is_no_witness"] = (df["witnesses"] == 0).astype(int)
    out["incident_severity_ordinal"] = df["incident_severity"].map(SEVERITY_ORDINAL).fillna(1).astype(int)
    out["is_major_damage"] = (df["incident_severity"] == "Major Damage").astype(int)

    # --- Flagged-risky features (module docstring points 1 & 2) ---
    # SH-02 / D3: gated behind include_proxy_features — OFF (the default)
    # means these two columns are not built at all, not merely masked to
    # zero, so a shipped OFF model genuinely never sees this signal.
    if include_proxy_features:
        out["is_highrisk_hobby"] = df["insured_hobbies"].isin(HIGH_RISK_HOBBIES).astype(int)
        out["is_exec_occupation"] = (df["insured_occupation"] == "exec-managerial").astype(int)

    # PB-02: no zip3-derived feature is built anymore — see module
    # docstring. `insured_zip` stays an EDA-only raw column.

    # --- Numeric passthrough ---
    for col in NUMERIC_PASSTHROUGH_COLUMNS:
        out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # --- One-hot encode categoricals (incl. insured_hobbies) ---
    cat_df = df[CATEGORICAL_COLUMNS].astype(str).copy()
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
