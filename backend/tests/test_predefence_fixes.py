"""test_predefence_fixes.py — regression tests for the pre-defence fixes
(EX-01 police-report label, MS-02 is_no_witness removal, TC-01 derived
total, DS-01 decision-support wording, MS-01 leak-free selection)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.feature_engineering import apply_missing_defaults, engineer_features
from app.ml.risk_policy import DECISION_SUPPORT_NOTICE, RECOMMENDED_ACTIONS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED = PROJECT_ROOT / "data" / "processed"


def _service():
    if not (MODELS_DIR / "feature_columns.json").exists():
        pytest.skip("model artifacts not built")
    from app.ml.inference import FraudScoringService
    return FraudScoringService.instance()


# ---- EX-01 -----------------------------------------------------------------
@pytest.mark.parametrize("answer", ["NO", "YES"])
def test_police_report_reason_shows_the_answer_actually_given(answer):
    """The bug: selecting NO produced a reason labelled 'police report
    available YES (0)'. Now exactly one police-report reason exists and it
    carries the claim's real answer."""
    svc = _service()
    X_scaled, X_raw = svc._prepare(pd.DataFrame([{"police_report_available": answer}]), return_raw=True)
    reasons = svc.explainer.top_reasons(X_scaled, k=200, X_row_raw=X_raw)
    police = [r for r in reasons if r["feature"] == "police_report_available"]
    assert len(police) == 1
    assert police[0]["value"] == answer
    other = "YES" if answer == "NO" else "NO"
    assert f": {other}" not in police[0]["sentence"]


def test_one_hot_reasons_are_grouped_and_additive():
    """Grouping must not lose SHAP mass: grouped contributions sum to the
    same total as the raw per-column SHAP values."""
    svc = _service()
    X_scaled, X_raw = svc._prepare(pd.DataFrame([{"incident_state": "SC", "collision_type": "Side Collision"}]), return_raw=True)
    reasons = svc.explainer.top_reasons(X_scaled, k=500, X_row_raw=X_raw)
    raw = svc.explainer.shap_values_for(X_scaled)[0]
    assert sum(r["shap_value"] for r in reasons) == pytest.approx(float(raw.sum()), abs=1e-9)
    assert len({r["feature"] for r in reasons}) == len(reasons)


# ---- MS-02 -----------------------------------------------------------------
def test_is_no_witness_is_gone_but_witness_count_remains():
    X = engineer_features(pd.DataFrame([{"witnesses": 0}]))
    assert "is_no_witness" not in X.columns
    assert "witnesses" in X.columns


# ---- TC-01 -----------------------------------------------------------------
def test_total_claim_amount_is_derived_from_components():
    df = apply_missing_defaults(pd.DataFrame([
        {"injury_claim": 1000, "property_claim": 2000, "vehicle_claim": 3000},
        {"injury_claim": 1000, "property_claim": 2000, "vehicle_claim": 3000, "total_claim_amount": 99},
        {"total_claim_amount": 500},
    ]))
    assert df["total_claim_amount"].tolist() == [6000, 6000, 500]


def test_training_data_total_always_equals_components():
    df = pd.read_csv(PROJECT_ROOT / "data" / "cleaned" / "insurance_claims_cleaned.csv")
    diff = df["total_claim_amount"] - df[["injury_claim", "property_claim", "vehicle_claim"]].sum(axis=1)
    assert (diff.abs() < 1e-6).all()


def test_api_schema_rejects_inconsistent_total():
    from pydantic import ValidationError
    from app.api.schemas import ClaimPayload
    ClaimPayload(injury_claim=1, property_claim=2, vehicle_claim=3, total_claim_amount=6)
    with pytest.raises(ValidationError):
        ClaimPayload(injury_claim=1, property_claim=2, vehicle_claim=3, total_claim_amount=600)


# ---- DS-01 -----------------------------------------------------------------
def test_no_action_text_claims_automatic_approval():
    for text in list(RECOMMENDED_ACTIONS.values()) + [DECISION_SUPPORT_NOTICE]:
        assert "auto-approv" not in text.lower()
    assert all("recommend" in a.lower() for a in RECOMMENDED_ACTIONS.values())


def test_score_one_returns_decision_support_notice():
    svc = _service()
    out = svc.score_one({"incident_severity": "Minor Damage"})
    assert out["decision_support_notice"] == DECISION_SUPPORT_NOTICE


# ---- MS-01 -----------------------------------------------------------------
def test_selection_artifacts_never_used_the_test_split():
    path = PROCESSED / "model_selection_folds.csv"
    if not path.exists():
        pytest.skip("selection artifacts not built")
    folds = pd.read_csv(path)
    with open(MODELS_DIR / "metrics.json") as f:
        m = json.load(f)
    # every outer fold trains+validates on the training split only
    assert (folds["n_train"] + folds["n_val"] == m["n_train"]).all()
    assert set(folds["model"]) >= {"major_damage_rule", "random_forest", "logistic_regression", "xgboost"}


def test_rule_baseline_flags_exactly_major_damage():
    from app.ml.model_selection import MajorDamageRule
    X = np.array([[0.0], [1.0], [0.0], [1.0]])
    rule = MajorDamageRule(col_index=0).fit(X, [0, 1, 0, 1])
    assert rule.predict_proba(X)[:, 1].tolist() == [0, 1, 0, 1]


def test_corrected_ttest_is_more_conservative_than_naive():
    from scipy import stats
    from app.ml.model_selection import corrected_resampled_ttest
    rng = np.random.default_rng(0)
    d = rng.normal(0.01, 0.02, 15)
    _, p_corr = corrected_resampled_ttest(d, n_train=640, n_val=160)
    _, p_naive = stats.ttest_1samp(d, 0)
    assert p_corr > p_naive
