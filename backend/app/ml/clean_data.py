"""
clean_data.py — cleans the raw 1,000-row insurance_claims dataset into
insurance_claims_cleaned.csv, mirroring the original FYP notebook's
documented treatment (01_eda.ipynb / 02_preprocessing.ipynb):

  - '?' is a missing-value sentinel used by three columns (`collision_type`,
    `property_damage`, `police_report_available`) in this dataset's own
    convention, not a real category — imputed with the mode, logged.
  - `_c39` (an all-null trailing artifact column some re-uploads of this
    dataset carry) is dropped if present.
  - `policy_number` is dropped as a pure identifier (no predictive value,
    would leak nothing but is dead weight).

SH-01 (fixed): `authorities_contacted` has 91 rows whose real value is the
literal string "None" — meaning "no authority was contacted", a genuine
category, same shape as "Police"/"Fire"/"Other"/"Ambulance". A previous
version of this module read the CSV with pandas' default `read_csv`
settings, under which "None" is one of pandas' own default NA-sentinel
strings, so those 91 rows silently became real NaNs and were then
mode-imputed to "Police" — fabricating "police was contacted" for 91
claims that actually said no authority was contacted at all, and
discarding what may be a genuine fraud signal (fraud claims disproportionately
skip involving authorities). Fixed by reading the raw CSV with
`keep_default_na=False, na_values=[""]` so only true empty cells are
treated as missing; "None" now survives as its own category and flows
through to `feature_engineering.py`'s one-hot encoding of
`authorities_contacted` as `authorities_contacted_None` like any other
level. With this fix, no column in the raw dataset has any genuine NaN at
all (verified directly against `insurance_claims_raw.csv`), so the
generic "genuine NaNs" mode-imputation loop below is now dead code for
this dataset — kept as a safety net for a future re-upload that might
actually contain missing cells, but it currently does nothing.

Produces two audit artifacts, same as the original project:
  - data/cleaned/cleaning_log.csv        — one row per cleaning action taken
  - data/cleaned/missing_value_treatment_plan.csv — per-column plan/rationale
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "insurance_claims_raw.csv"
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"

QUESTION_MARK_COLUMNS = ["collision_type", "property_damage", "police_report_available"]


def clean() -> pd.DataFrame:
    # SH-01: keep_default_na=False + explicit na_values=[""] so the literal
    # string "None" in `authorities_contacted` (a genuine "no authority
    # contacted" category) is NOT swallowed by pandas' default NA-sentinel
    # list, which otherwise treats "None" as missing data. See module
    # docstring.
    df = pd.read_csv(RAW_PATH, keep_default_na=False, na_values=[""])
    log_rows = []
    plan_rows = []

    # Drop all-null artifact column some re-uploads carry.
    for junk_col in [c for c in df.columns if c.startswith("_c") or c.startswith("Unnamed")]:
        if df[junk_col].isna().all():
            df = df.drop(columns=[junk_col])
            log_rows.append({"action": "drop_column", "column": junk_col, "reason": "all-null artifact column"})

    # '?' sentinel columns -> mode imputation.
    for col in QUESTION_MARK_COLUMNS:
        n_qmark = int((df[col] == "?").sum())
        mode_val = df.loc[df[col] != "?", col].mode().iloc[0]
        df[col] = df[col].replace("?", mode_val)
        log_rows.append({
            "action": "impute_sentinel", "column": col, "sentinel": "?",
            "n_affected": n_qmark, "fill_value": mode_val,
        })
        plan_rows.append({
            "column": col, "missing_marker": "?", "n_missing": n_qmark,
            "strategy": "mode imputation", "rationale": "categorical field; '?' is this dataset's own missing-value sentinel, not a real level",
        })

    # Genuine NaNs.
    for col in df.columns[df.isna().any()]:
        n_na = int(df[col].isna().sum())
        mode_val = df[col].mode().iloc[0]
        df[col] = df[col].fillna(mode_val)
        log_rows.append({"action": "impute_na", "column": col, "n_affected": n_na, "fill_value": mode_val})
        plan_rows.append({
            "column": col, "missing_marker": "NaN", "n_missing": n_na,
            "strategy": "mode imputation", "rationale": "categorical field, low missingness rate (<10%)",
        })

    # Drop pure identifier.
    if "policy_number" in df.columns:
        df = df.drop(columns=["policy_number"])
        log_rows.append({"action": "drop_column", "column": "policy_number", "reason": "pure identifier, no predictive value"})

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED_DIR / "insurance_claims_cleaned.csv", index=False)
    pd.DataFrame(log_rows).to_csv(CLEANED_DIR / "cleaning_log.csv", index=False)
    pd.DataFrame(plan_rows).to_csv(CLEANED_DIR / "missing_value_treatment_plan.csv", index=False)

    print(f"Cleaned {len(df)} rows x {len(df.columns)} cols -> {CLEANED_DIR / 'insurance_claims_cleaned.csv'}")
    print(f"Fraud rate: {(df['fraud_reported'] == 'Y').mean():.1%}")
    return df


if __name__ == "__main__":
    clean()
