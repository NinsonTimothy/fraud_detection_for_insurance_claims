"""test_explainer.py — regression coverage for PB-05.

Two bugs/gaps fixed together:

1. ClaimExplainer used to unconditionally build shap.TreeExplainer(model)
   regardless of model type. Reproduced directly against the actually
   shipped Logistic Regression challenger artifact
   (models/logistic_regression_final.pkl):

       >>> shap.TreeExplainer(logistic_regression_estimator)
       shap.utils._exceptions.InvalidModelError: Model type not yet
       supported by TreeExplainer: <class 'sklearn.linear_model._logistic.LogisticRegression'>

   Never hit in practice (ClaimExplainer was only ever instantiated with
   the RF estimator), but a real, unexercised bug. Fixed by routing to
   the correct SHAP explainer class per model type.

2. top_reasons() hardcoded k=3 everywhere it was called; the new default
   is 8 (explainer.DEFAULT_TOP_K), and each reason dict is now genuinely
   plain-language (rank/display_name/direction/impact), not just a
   feature name and a raw signed SHAP float.
"""
import json
import sys
from pathlib import Path

import joblib
from app.ml.inference import underlying_estimator
import numpy as np
import pandas as pd
import pytest
import shap
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.explainer import DEFAULT_TOP_K, ClaimExplainer, _format_value, _impact_label

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"


def _artifacts_available() -> bool:
    return (MODELS_DIR / "random_forest_final.pkl").exists() and (MODELS_DIR / "logistic_regression_final.pkl").exists()


@pytest.fixture(scope="module")
def artifacts():
    if not _artifacts_available():
        pytest.skip("model artifacts not built yet — run `python -m app.ml.train` first")
    with open(MODELS_DIR / "feature_columns.json") as f:
        feature_columns = json.load(f)
    rf_pipeline = joblib.load(MODELS_DIR / "random_forest_final.pkl")
    lr_pipeline = joblib.load(MODELS_DIR / "logistic_regression_final.pkl")
    test_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "risk_scores_test.csv")
    X_test_scaled = test_df[feature_columns]
    return {
        "feature_columns": feature_columns,
        "rf": underlying_estimator(rf_pipeline),
        "lr": underlying_estimator(lr_pipeline),
        "X_test_scaled": X_test_scaled,
    }


def test_treeexplainer_cannot_explain_logistic_regression_directly():
    """Locks in the reproduction of the original bug's root cause: SHAP's
    TreeExplainer itself, unconditionally applied, genuinely cannot handle
    a linear model — this is why ClaimExplainer must route by type."""
    lr = LogisticRegression().fit(np.random.default_rng(0).normal(size=(20, 3)), [0, 1] * 10)
    with pytest.raises(Exception):  # shap.utils._exceptions.InvalidModelError
        shap.TreeExplainer(lr)


def test_claim_explainer_routes_random_forest_to_tree_explainer(artifacts):
    explainer = ClaimExplainer(artifacts["rf"], artifacts["feature_columns"])
    assert explainer.explainer_kind == "tree"
    sv = explainer.shap_values_for(artifacts["X_test_scaled"].iloc[:5])
    assert sv.shape == (5, len(artifacts["feature_columns"]))


def test_claim_explainer_requires_background_for_logistic_regression(artifacts):
    with pytest.raises(ValueError):
        ClaimExplainer(artifacts["lr"], artifacts["feature_columns"])  # no background_data


def test_claim_explainer_routes_logistic_regression_to_linear_explainer(artifacts):
    """PB-05: this is the case that used to crash before the fix — a
    Logistic Regression estimator, given a background sample, now
    produces valid per-row SHAP values instead of raising InvalidModelError."""
    background = artifacts["X_test_scaled"].sample(30, random_state=42)
    explainer = ClaimExplainer(artifacts["lr"], artifacts["feature_columns"], background_data=background)
    assert explainer.explainer_kind == "linear"
    sv = explainer.shap_values_for(artifacts["X_test_scaled"].iloc[:5])
    assert sv.shape == (5, len(artifacts["feature_columns"]))
    assert np.all(np.isfinite(sv))


def test_top_reasons_default_k_is_eight(artifacts):
    explainer = ClaimExplainer(artifacts["rf"], artifacts["feature_columns"])
    row_scaled = artifacts["X_test_scaled"].iloc[[0]]
    reasons = explainer.top_reasons(row_scaled)
    assert len(reasons) == DEFAULT_TOP_K == 8


def test_top_reasons_dict_is_plain_language(artifacts):
    explainer = ClaimExplainer(artifacts["rf"], artifacts["feature_columns"])
    row_scaled = artifacts["X_test_scaled"].iloc[[0]]
    reasons = explainer.top_reasons(row_scaled, k=5)
    assert len(reasons) == 5
    ranks = [r["rank"] for r in reasons]
    assert ranks == [1, 2, 3, 4, 5]
    for r in reasons:
        assert r["direction"] in {"increased", "decreased"}
        assert r["impact"] in {"strongly", "moderately", "slightly"}
        # EX-01: display names are curated labels now, not humanized column names.
        from app.ml.explainer import FEATURE_LABELS, FEATURE_LABELS_EXTRA
        assert r["display_name"] == FEATURE_LABELS_EXTRA.get(r["feature"]) or r["display_name"] == FEATURE_LABELS.get(r["feature"], r["feature"].replace("_", " ").replace("-", " ").strip())
        assert r["display_name"] in r["sentence"]
    # The rank-1 reason must be the single largest-magnitude SHAP driver.
    assert abs(reasons[0]["shap_value"]) == max(abs(r["shap_value"]) for r in reasons)


def test_top_reasons_uses_raw_unscaled_values_when_given(artifacts):
    explainer = ClaimExplainer(artifacts["rf"], artifacts["feature_columns"])
    row_scaled = artifacts["X_test_scaled"].iloc[[0]]
    # A fabricated "raw" row with an obviously human-scale age, distinct
    # from whatever z-score sits in row_scaled.
    row_raw = row_scaled.copy()
    row_raw["age"] = 41
    reasons = explainer.top_reasons(row_scaled, k=len(artifacts["feature_columns"]), X_row_raw=row_raw)
    age_reason = next(r for r in reasons if r["feature"] == "age")
    assert age_reason["value"] == pytest.approx(41.0)
    assert "41" in age_reason["sentence"]


def test_top_reasons_batch_matches_per_row_top_reasons(artifacts):
    """PB-18: top_reasons_batch() must produce EXACTLY what calling
    top_reasons() once per row would — it's an efficiency refactor (one
    SHAP call over the matrix instead of N), not a different computation.
    This is the test that would catch the refactor silently changing
    output (e.g. a row/column indexing slip in _reasons_for_row)."""
    explainer = ClaimExplainer(artifacts["rf"], artifacts["feature_columns"])
    rows_scaled = artifacts["X_test_scaled"].iloc[:5]

    batch_reasons = explainer.top_reasons_batch(rows_scaled)
    assert len(batch_reasons) == 5

    for i in range(5):
        single_reasons = explainer.top_reasons(rows_scaled.iloc[[i]])
        assert batch_reasons[i] == single_reasons


def test_top_reasons_batch_respects_k(artifacts):
    explainer = ClaimExplainer(artifacts["rf"], artifacts["feature_columns"])
    rows_scaled = artifacts["X_test_scaled"].iloc[:3]
    batch_reasons = explainer.top_reasons_batch(rows_scaled, k=4)
    assert all(len(reasons) == 4 for reasons in batch_reasons)


def test_top_reasons_batch_uses_raw_unscaled_values_when_given(artifacts):
    explainer = ClaimExplainer(artifacts["rf"], artifacts["feature_columns"])
    rows_scaled = artifacts["X_test_scaled"].iloc[:3]
    rows_raw = rows_scaled.copy()
    rows_raw["age"] = [22, 41, 63]

    batch_reasons = explainer.top_reasons_batch(rows_scaled, k=len(artifacts["feature_columns"]), X_raw=rows_raw)
    for i, expected_age in enumerate([22, 41, 63]):
        age_reason = next(r for r in batch_reasons[i] if r["feature"] == "age")
        assert age_reason["value"] == pytest.approx(float(expected_age))


def test_impact_label_relative_to_row_max():
    assert _impact_label(9.0, 10.0) == "strongly"
    assert _impact_label(5.0, 10.0) == "moderately"
    assert _impact_label(1.0, 10.0) == "slightly"
    assert _impact_label(0.0, 0.0) == "slightly"  # degenerate all-zero row, no crash


def test_format_value_flag_vs_numeric():
    # A genuine flag (column dtype bool, or forced via is_flag=True) reads
    # yes/no; a genuine numeric feature that happens to equal 0 or 1 (e.g.
    # witnesses=1) must NOT be relabeled yes/no just because of its value.
    assert _format_value(True, is_flag=True) == "yes"
    assert _format_value(False, is_flag=True) == "no"
    assert _format_value(1, is_flag=False) == "1"
    assert _format_value(0, is_flag=False) == "0"
    assert _format_value(2.5) == "2.50"
    assert _format_value(3.0) == "3"
