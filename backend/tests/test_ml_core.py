"""test_ml_core.py — regression test for the single-row-vs-batch scoring
bug found during this build: naive pd.get_dummies on a 1-row frame only
emits dummy columns for that row's own category, and reindexing without an
explicit fill_value turns every other category into NaN ("missing")
instead of 0 ("not this category")."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.feature_engineering import align_to_training_columns, engineer_features


@pytest.fixture(scope="module")
def sample_data():
    df = pd.read_csv(Path(__file__).resolve().parents[2] / "data" / "cleaned" / "insurance_claims_cleaned.csv")
    y = (df["fraud_reported"] == "Y").astype(int)
    return df, y


def test_single_row_scoring_matches_batch_scoring(sample_data):
    df, y = sample_data
    X_batch = engineer_features(df)

    single_row = df.iloc[[3]].reset_index(drop=True)
    X_single = engineer_features(single_row)
    X_single_aligned = align_to_training_columns(X_single, list(X_batch.columns))

    # No NaN — every category this row doesn't belong to should read 0.
    assert X_single_aligned.isna().sum().sum() == 0
    # The row's own values should match the equivalent row computed in a full batch.
    expected = X_batch.iloc[[3]][X_single_aligned.columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(X_single_aligned.reset_index(drop=True), expected, check_dtype=False)


def test_align_to_training_columns_fills_missing_with_zero_not_nan():
    df = pd.DataFrame({"a": [1], "b": [2]})
    aligned = align_to_training_columns(df, ["a", "b", "c", "d"])
    assert aligned["c"].iloc[0] == 0
    assert aligned["d"].iloc[0] == 0
    assert aligned.isna().sum().sum() == 0


def test_engineer_features_no_nans_on_full_dataset(sample_data):
    df, y = sample_data
    X = engineer_features(df)
    assert X.isna().sum().sum() == 0


def test_no_zip3_derived_columns_survive_feature_engineering(sample_data):
    """Regression test for PB-02: insured_zip // 100 on 6-digit US ZIPs is a
    4-digit prefix, not a genuine 3-digit ZIP3, and was near-row-unique
    (515 groups from 1,000 rows). The target-encoded zip3_risk_tier_* block
    built from it let the model memorize labels on train while collapsing
    to a flat rate on test, and consumed 53.2% of total SHAP weight. Fixed
    by removing the zip3-derived feature entirely."""
    df, y = sample_data
    X = engineer_features(df)
    assert not any(col.startswith("zip3") for col in X.columns)


def test_missing_raw_columns_fall_back_to_documented_defaults():
    from app.ml.feature_engineering import MISSING_COLUMN_DEFAULTS, apply_missing_defaults
    sparse = pd.DataFrame({"age": [40]})
    filled = apply_missing_defaults(sparse)
    for col, default in MISSING_COLUMN_DEFAULTS.items():
        assert col in filled.columns
        if col != "age":
            assert filled[col].iloc[0] == default
