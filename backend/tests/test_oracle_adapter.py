"""test_oracle_adapter.py — PB-16: `oracle_adapter.py` (docs/REBUILD_NOTES.md
§'Generalization testing') had no test coverage at all before this ticket,
despite being the sole mechanism behind this project's one genuine
out-of-sample validation number (ROC-AUC 0.496 on Oracle — see
docs/generalization_and_cv_results.md). Its own module docstring makes
specific, checkable claims: exactly 9 of 35 raw fields are mapped for
real, everything else is left absent for `apply_missing_defaults()` to
fill, and every mapped field uses a documented, disclosed transform (not
an invented one). This file locks those claims in against the real
`data/external/oracle/fraud_oracle.csv` file already in the repo, so a
future edit to the adapter can't silently drift from what the docs say
it does without a test failing."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.feature_engineering import RAW_FEATURE_COLUMNS, engineer_features
from app.ml.oracle_adapter import load_oracle_raw, map_oracle_to_raw_schema

# The 9 fields oracle_adapter.py's own module docstring documents as
# genuinely mapped — see "Fields mapped for real (9 of 35..." there.
DOCUMENTED_MAPPED_FIELDS = {
    "age", "insured_sex", "policy_deductable", "police_report_available",
    "witnesses", "number_of_vehicles_involved", "incident_date",
    "auto_year", "policy_bind_date",
}


@pytest.fixture(scope="module")
def oracle_raw():
    return load_oracle_raw()


def test_load_oracle_raw_matches_the_known_row_and_column_shape(oracle_raw):
    """A basic sanity/drift guard on the source file itself — if this
    dataset is ever swapped or truncated, every other test in this file
    would otherwise fail with confusing downstream errors instead of a
    clear "the source file changed" signal."""
    assert len(oracle_raw) == 15420
    assert "FraudFound_P" in oracle_raw.columns


def test_mapped_columns_match_the_documented_field_list_exactly(oracle_raw):
    """The module docstring is explicit that exactly 9 fields are mapped
    for real and lists them by name (a PB-21 fix corrected an earlier
    miscount) — this must not silently grow or shrink without the
    docstring being updated to match."""
    mapped, _ = map_oracle_to_raw_schema(oracle_raw)
    assert set(mapped.columns) == DOCUMENTED_MAPPED_FIELDS
    assert len(mapped.columns) == 9


def test_unmapped_fields_are_genuinely_absent_not_defaulted_here(oracle_raw):
    """The adapter's honesty property: everything NOT genuinely derivable
    from Oracle must be left OUT of its output entirely, so
    `apply_missing_defaults()` — not the adapter itself — is what fills
    it with a disclosed, documented constant. If the adapter started
    inventing values for e.g. `incident_severity` (which Oracle has no
    equivalent for at all), the external-validation numbers in
    docs/generalization_and_cv_results.md would no longer mean what they
    claim to mean."""
    mapped, _ = map_oracle_to_raw_schema(oracle_raw)
    unmapped = set(RAW_FEATURE_COLUMNS) - DOCUMENTED_MAPPED_FIELDS
    assert unmapped, "sanity check: there should be unmapped fields to assert about"
    assert not (unmapped & set(mapped.columns))


def test_witness_present_maps_losslessly_to_zero_one(oracle_raw):
    mapped, _ = map_oracle_to_raw_schema(oracle_raw)
    assert set(mapped["witnesses"].unique()) <= {0, 1}
    # Cross-check against the raw source column directly, not just the
    # output's shape — a real Yes -> 1 and No -> 0, not e.g. both to 0.
    yes_rows = oracle_raw["WitnessPresent"] == "Yes"
    assert (mapped.loc[yes_rows, "witnesses"] == 1).all()
    assert (mapped.loc[~yes_rows, "witnesses"] == 0).all()


def test_sex_and_police_report_are_uppercased_to_match_this_schemas_vocabulary(oracle_raw):
    """This project's own schema uses "MALE"/"FEMALE" and "YES"/"NO"
    (see schemas.py's ClaimPayload); Oracle's source columns are
    "Male"/"Female" and "Yes"/"No" — a case mismatch here would silently
    fail every downstream categorical check that does an exact string
    comparison against this schema's vocabulary."""
    mapped, _ = map_oracle_to_raw_schema(oracle_raw)
    assert set(mapped["insured_sex"].unique()) <= {"MALE", "FEMALE"}
    assert set(mapped["police_report_available"].unique()) <= {"YES", "NO"}


def test_zero_age_sentinel_is_replaced_not_passed_through(oracle_raw):
    """Oracle uses 0 as its own missing-age sentinel (320 of 15,420 rows,
    verified directly) — passing that straight through would make the
    model see a literal age of 0, a data-quality artifact, not a genuine
    input. The adapter replaces it with the median of Oracle's own
    real (>0) ages."""
    assert (oracle_raw["Age"] == 0).sum() > 0  # the sentinel genuinely exists in this data
    mapped, _ = map_oracle_to_raw_schema(oracle_raw)
    assert (mapped["age"] > 0).all()


def test_mapped_oracle_rows_pass_through_engineer_features_with_no_nans(oracle_raw):
    """The whole point of the adapter is that Oracle rows can flow through
    the EXACT SAME `engineer_features()` pipeline the live model uses.
    If any mapped/defaulted combination ever produced a NaN here, every
    Oracle-scored row downstream would silently get corrupted features
    (the same failure class PB-20/`test_ml_core.py` already guards
    against for the live API's own inputs)."""
    mapped, y_true = map_oracle_to_raw_schema(oracle_raw)
    X = engineer_features(mapped)
    assert X.isna().sum().sum() == 0
    assert len(X) == len(oracle_raw)
    assert set(y_true.unique()) <= {0, 1}


def test_incident_severity_derived_features_are_constant_across_oracle_rows(oracle_raw):
    """A specific, checkable instance of the adapter's central honesty
    claim: Oracle has no `incident_severity` equivalent at all, so every
    row must fall back to the SAME documented default
    (`MISSING_COLUMN_DEFAULTS["incident_severity"]` = "Minor Damage"),
    making `is_major_damage` constant for every Oracle row. This is
    exactly the generalization gap docs/insurance_fyp_project_handoff.md
    quantifies (93.2% of SHAP weight sits on features constant on
    Oracle-mapped data) — this test locks in the underlying mechanism,
    not just the downstream ROC-AUC number."""
    mapped, _ = map_oracle_to_raw_schema(oracle_raw)
    X = engineer_features(mapped)
    assert "is_major_damage" in X.columns
    assert X["is_major_damage"].nunique() == 1
