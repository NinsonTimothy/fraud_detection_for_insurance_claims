"""test_psi.py — regression coverage for PB-10 (two bugs found together):

1. Scale mismatch: evaluate_oracle.py used to compare risk_scores_test.csv
   (StandardScaler-SCALED test features, z-scores) against X_oracle
   (UNSCALED engineered features) via PSI. Reproduced directly: PSI("age")
   scaled-vs-unscaled = 6.919 (nonsensically large — PSI is conventionally
   read on a 0-~2 scale), vs. 0.048 ("no significant shift", the honest
   answer) comparing unscaled-vs-unscaled. Fixed by adding
   psi_reference_features.csv (train.py) — an UNSCALED snapshot of the test
   features — and pointing evaluate_oracle.py at it instead.

2. Boolean-dtype crash: fixing (1) exposed a second, previously-latent bug.
   pandas 3.0.2's pd.get_dummies() emits bool-dtype one-hot columns (not
   the historical uint8/int8); these survive a CSV round-trip as bool.
   pd.api.types.is_numeric_dtype() returns True for bool, so psi_report()
   routed them into psi_numeric() -> _bucket_edges() -> .quantile(), and
   numpy's quantile interpolation crashes on boolean arrays:
   "TypeError: numpy boolean subtract, the `-` operator, is not
   supported...". This was masked before fix (1) because the old
   (buggy) reference, risk_scores_test.csv, was StandardScaler output —
   always float64, so .quantile() was never called on a bool Series.
   Fixed in psi.py by excluding bool dtype from the numeric branch and
   routing it to psi_categorical() instead (semantically correct: a 0/1
   flag is a category, not a continuous quantity).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.psi import psi_categorical, psi_numeric, psi_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_psi_numeric_same_distribution_is_near_zero():
    rng = np.random.default_rng(42)
    reference = pd.Series(rng.normal(size=2000))
    comparison = pd.Series(rng.normal(size=2000))
    assert psi_numeric(reference, comparison) < 0.05


def test_psi_numeric_shifted_distribution_is_flagged():
    rng = np.random.default_rng(42)
    reference = pd.Series(rng.normal(loc=0, size=2000))
    comparison = pd.Series(rng.normal(loc=5, size=2000))
    assert psi_numeric(reference, comparison) >= 0.2


def test_psi_report_does_not_crash_on_boolean_columns():
    """PB-10: this used to raise TypeError: numpy boolean subtract..."""
    reference_df = pd.DataFrame({"flag": pd.Series([True, False, True, True, False] * 40)})
    comparison_df = pd.DataFrame({"flag": pd.Series([True, True, True, False, False] * 40)})
    report = psi_report(reference_df, comparison_df, ["flag"])
    assert len(report) == 1
    assert np.isfinite(report.loc[0, "psi"])


def test_psi_report_routes_bool_columns_to_categorical_not_numeric():
    reference_df = pd.DataFrame({"flag": pd.Series([True, False] * 50)})
    comparison_df = pd.DataFrame({"flag": pd.Series([True, False] * 50)})
    report_via_psi_report = psi_report(reference_df, comparison_df, ["flag"])[["psi"]].iloc[0, 0]
    expected_categorical = psi_categorical(
        reference_df["flag"].astype(str), comparison_df["flag"].astype(str)
    )
    assert report_via_psi_report == pytest.approx(expected_categorical)


def test_psi_reference_features_csv_is_unscaled():
    """PB-10 (scale-mismatch): the dedicated PSI-comparison artifact must
    hold RAW engineered feature values, not StandardScaler z-scores — a
    z-scored 'age' would show mean~0/std~1; a real one is a human age."""
    path = PROJECT_ROOT / "data" / "processed" / "psi_reference_features.csv"
    if not path.exists():
        pytest.skip("psi_reference_features.csv not generated yet — run `python -m app.ml.train` first")
    df = pd.read_csv(path)
    assert "age" in df.columns
    assert df["age"].min() >= 0
    assert df["age"].max() > 10  # a z-scored age would never reach this
    assert df["age"].mean() > 5  # z-scored age has mean ~0


def test_psi_reference_features_csv_has_bool_onehot_columns():
    """EX-02 changed one-hot dummies from bool to int 0/1; the original
    bool round-trip bug is still covered by casting a column to bool here."""
    """Confirms the fixture this bug actually lives in: one-hot columns
    round-trip through CSV as bool dtype in this pandas version, and
    psi_report() must handle that without crashing (see test above)."""
    path = PROJECT_ROOT / "data" / "processed" / "psi_reference_features.csv"
    if not path.exists():
        pytest.skip("psi_reference_features.csv not generated yet — run `python -m app.ml.train` first")
    df = pd.read_csv(path)
    onehot = [c for c in df.columns if c.startswith(("police_report_available_", "insured_sex_"))]
    assert onehot and all(set(df[c].unique()) <= {0, 1} for c in onehot)
    df = df.assign(**{c: df[c].astype(bool) for c in onehot})
    bool_cols = onehot
    # And psi_report must not crash when asked to compare a bool column
    # against itself.
    report = psi_report(df, df, bool_cols[:3])
    assert len(report) == len(bool_cols[:3])
