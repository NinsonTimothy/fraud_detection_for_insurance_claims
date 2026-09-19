"""
oracle_adapter.py — maps the Oracle auto-insurance-fraud dataset (a real,
independently-collected dataset, ~15,420 US claims from the early-mid
1990s) onto this project's own 35-column raw schema, so the SAME
`feature_engineering.engineer_features()` pipeline the live model uses can
score it — an honest generalization test, not a fresh model trained on
Oracle's own fields.

Same rule as the sibling MoMo Guard project's `paysim_adapter.py`: map only
genuinely-overlapping fields for real; everything else is left OUT so
`feature_engineering.apply_missing_defaults()` handles it via its own
documented, disclosed fallback constants. Nothing is invented to make the
model look better or worse on Oracle than it actually is.

Fields mapped for real (9 of 35 — count matches the 9 `out[...] =`
assignments in map_oracle_to_raw_schema() below; PB-21: an earlier draft
of this docstring said "10 of 35" and counted total_claim_amount as
mapped, which it explicitly is not — see below):
  age                        <- Age
  insured_sex                <- Sex
  policy_deductable           <- Deductible
  police_report_available     <- PoliceReportFiled
  witnesses                   <- WitnessPresent (LOSSY: Yes/No -> 1/0, not a
                                  real count — Oracle has no witness count)
  number_of_vehicles_involved <- NumberOfCars (bucketed string -> numeric)
  incident_date                <- reconstructed from Year + MonthClaimed +
                                  WeekOfMonthClaimed (approximate day-of-month)
  auto_year                    <- incident_date.year - AgeOfVehicle (bucketed)
  policy_bind_date              <- incident_date - Days_Policy_Claim (bucketed)

Notably NOT mapped — total_claim_amount: Oracle has no claim-dollar fields
at all (see "Fields Oracle has no equivalent for" below), so it's left as
fallback, which also means claim_to_premium_ratio, vehicle_claim_pct,
injury_claim_pct, property_claim_pct all collapse to their neutral
fallback for every Oracle row.

Fields Oracle has NO equivalent for at all (left to fallback):
  months_as_customer, policy_state, policy_csl, policy_annual_premium,
  umbrella_limit, insured_zip, insured_education_level, insured_occupation,
  insured_hobbies, insured_relationship, capital-gains, capital-loss,
  incident_type, collision_type, incident_severity, authorities_contacted,
  incident_state, incident_hour_of_the_day, property_damage,
  total_claim_amount, injury_claim, property_claim, vehicle_claim,
  auto_make (auto_make IS mappable via `Make`, but is unused by the model
  — see feature_engineering.py's CATEGORICAL_COLUMNS note — so mapping it
  would have zero effect and is skipped)

This means `incident_severity`/`is_major_damage`-derived features are
constant for every Oracle row (Oracle has no equivalent field, so they all
fall back to the same documented default). This is exactly the
generalization question this adapter exists to measure honestly, not paper
over.

PB-02 note: an earlier version of this docstring also named
`zip3_risk_tier` as constant-on-Oracle and as "the single largest share of
this model's SHAP weight" — that feature has since been removed entirely
(it was a near-row-unique target-encoding bug, not usable signal; see
feature_engineering.py's module docstring), so it no longer applies here.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ORACLE_RAW_PATH = PROJECT_ROOT / "data" / "external" / "oracle" / "fraud_oracle.csv"

NUM_CARS_MAP = {"1 vehicle": 1, "2 vehicles": 2, "3 to 4": 3, "5 to 8": 5, "more than 8": 9}
AGE_OF_VEHICLE_MAP = {"new": 0, "2 years": 2, "3 years": 3, "4 years": 4, "5 years": 5, "6 years": 6, "7 years": 7, "more than 7": 9}
DAYS_POLICY_CLAIM_MAP = {"none": 0, "8 to 15": 12, "15 to 30": 22, "more than 30": 45}
MONTH_MAP = {m: i + 1 for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}


def load_oracle_raw() -> pd.DataFrame:
    return pd.read_csv(ORACLE_RAW_PATH)


def map_oracle_to_raw_schema(oracle_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Returns (mapped_df, y_true) where mapped_df has only the columns
    genuinely derivable from Oracle — everything else is left absent for
    `feature_engineering.apply_missing_defaults()` to fill in."""
    df = oracle_df.copy()
    out = pd.DataFrame(index=df.index)

    out["age"] = df["Age"].replace(0, df["Age"][df["Age"] > 0].median())  # Oracle uses 0 as its own missing-age sentinel
    out["insured_sex"] = df["Sex"].str.upper()
    out["policy_deductable"] = df["Deductible"]
    out["police_report_available"] = df["PoliceReportFiled"].str.upper()
    out["witnesses"] = (df["WitnessPresent"] == "Yes").astype(int)  # lossy: Oracle has no witness count
    out["number_of_vehicles_involved"] = df["NumberOfCars"].map(NUM_CARS_MAP).fillna(1).astype(int)

    day_approx = (df["WeekOfMonthClaimed"].clip(upper=4) - 1) * 7 + 1
    month_num = df["MonthClaimed"].map(MONTH_MAP).fillna(1).astype(int)
    incident_date = pd.to_datetime(
        dict(year=df["Year"], month=month_num, day=day_approx.clip(lower=1, upper=28)),
        errors="coerce",
    )
    out["incident_date"] = incident_date.dt.strftime("%Y-%m-%d")

    veh_age_years = df["AgeOfVehicle"].map(AGE_OF_VEHICLE_MAP).fillna(5)
    out["auto_year"] = (incident_date.dt.year - veh_age_years).fillna(incident_date.dt.year.median()).astype(int)

    days_policy_claim = df["Days_Policy_Claim"].map(DAYS_POLICY_CLAIM_MAP).fillna(20)
    out["policy_bind_date"] = (incident_date - pd.to_timedelta(days_policy_claim, unit="D")).dt.strftime("%Y-%m-%d")

    y_true = (df["FraudFound_P"] == 1).astype(int)
    return out, y_true
