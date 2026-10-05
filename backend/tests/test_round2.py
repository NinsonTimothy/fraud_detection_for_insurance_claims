"""test_round2.py — pre-defence round 2 (D1 regression, D2 controlled
scenarios, D3 pipeline integrity). Artifact-dependent tests skip cleanly if
`python -m app.ml.run_all` has not been run."""
from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.feature_engineering import (CATEGORICAL_COLUMNS, MISSING_COLUMN_DEFAULTS, RAW_FEATURE_COLUMNS,
                                        engineer_features, inconsistent_total_mask)
from app.ml.form_spec import DERIVED_ON_FORM, DOCUMENTED_DEFAULTED, FORM_FIELDS

ROOT = Path(__file__).resolve().parents[2]
MODELS, PROC = ROOT / "models", ROOT / "data" / "processed"
needs_artifacts = pytest.mark.skipif(not (MODELS / "metrics.json").exists(), reason="run app.ml.run_all first")


def _svc():
    from app.ml.inference import FraudScoringService
    return FraudScoringService.instance()


# ============================== D1 =========================================
@needs_artifacts
@pytest.mark.parametrize("answer,shown", [("NO", "No"), ("YES", "Yes")])
def test_police_report_parent_label_and_true_value(answer, shown):
    r = _svc().score_one({**MISSING_COLUMN_DEFAULTS, "police_report_available": answer})
    allr = r["factors_increasing_risk"] + r["factors_reducing_risk"] + r["top_reasons"]
    svc = _svc()
    X, Xr = svc._prepare(pd.DataFrame([{**MISSING_COLUMN_DEFAULTS, "police_report_available": answer}]), return_raw=True)
    reasons = svc.explainer.top_reasons(X, k=500, X_row_raw=Xr)
    pol = [x for x in reasons if x["feature"] == "police_report_available"]
    assert len(pol) == 1 and pol[0]["display_value"] == shown and pol[0]["display_name"] == "Police report available"
    assert f"is {shown}," in pol[0]["sentence"]
    assert all("(0)" not in x["sentence"] for x in allr)


@needs_artifacts
def test_severity_is_one_reason_with_true_value():
    svc = _svc()
    X, Xr = svc._prepare(pd.DataFrame([{**MISSING_COLUMN_DEFAULTS, "incident_severity": "Major Damage"}]), return_raw=True)
    sev = [x for x in svc.explainer.top_reasons(X, k=500, X_row_raw=Xr) if x["feature"] == "incident_severity"]
    assert len(sev) == 1 and sev[0]["display_value"] == "Major Damage"


def test_witnesses_numeric_and_no_is_no_witness():
    X = engineer_features(pd.DataFrame([{**MISSING_COLUMN_DEFAULTS, "witnesses": 2}]))
    assert "is_no_witness" not in X.columns and float(X["witnesses"].iloc[0]) == 2.0


def test_witness_fraud_rate_pattern_documented_matches_data():
    df = pd.read_csv(ROOT / "data" / "cleaned" / "insurance_claims_cleaned.csv")
    rate = df.groupby("witnesses")["fraud_reported"].apply(lambda s: (s == "Y").mean())
    assert rate.loc[0] < rate.loc[2]  # the documented artefact: more witnesses, more fraud
    text = (ROOT / "docs" / "LIMITATIONS.md").read_text()
    for w in range(4):
        assert f"{rate.loc[w] * 100:.1f}%" in text


def test_inconsistent_total_mask():
    df = pd.DataFrame([{"total_claim_amount": 6, "injury_claim": 1, "property_claim": 2, "vehicle_claim": 3},
                       {"total_claim_amount": 6.5, "injury_claim": 1, "property_claim": 2, "vehicle_claim": 3},
                       {"total_claim_amount": 600, "injury_claim": 1, "property_claim": 2, "vehicle_claim": 3},
                       {"total_claim_amount": 600},
                       {"total_claim_amount": 100, "property_claim": 397, "vehicle_claim": 1},
                       {"total_claim_amount": 600, "property_claim": 397}])
    assert inconsistent_total_mask(df).tolist() == [False, False, True, False, True, False]


@needs_artifacts
def test_api_rejects_inconsistent_total_single_and_batch():
    from fastapi.testclient import TestClient
    from app.core.config import API_KEY
    from app.main import app
    c = TestClient(app, headers={"X-API-Key": API_KEY})
    ok = c.post("/score", json={"payload": {"injury_claim": 1, "property_claim": 2, "vehicle_claim": 3, "total_claim_amount": 6}})
    assert ok.status_code == 200  # same shape, consistent total -> accepted
    r = c.post("/score", json={"payload": {"injury_claim": 1, "property_claim": 2, "vehicle_claim": 3, "total_claim_amount": 600}})
    assert r.status_code == 422 and "total_claim_amount" in r.text  # rejected for the RIGHT reason
    csv = b"injury_claim,property_claim,vehicle_claim,total_claim_amount\n1,2,3,6\n1,2,3,600\n"
    r = c.post("/score/batch", files={"file": ("b.csv", io.BytesIO(csv), "text/csv")})
    body = r.json()
    assert r.status_code == 200 and body["n_scored"] == 1 and body["n_rejected"] == 1
    assert body["rejected_rows"][0]["row_index"] == 1 and "total_claim_amount" in body["rejected_rows"][0]["error"]


@needs_artifacts
def test_every_categorical_option_was_seen_in_training():
    from app.ml.explainer import build_source_map
    cols = json.load(open(MODELS / "feature_columns.json"))
    model_cats: dict[str, set] = {}
    for _, (src, cat) in build_source_map(cols).items():
        if cat is not None:
            model_cats.setdefault(src, set()).add(cat)
    df = pd.read_csv(ROOT / "data" / "cleaned" / "insurance_claims_cleaned.csv", keep_default_na=False, na_values=[""])
    for c in CATEGORICAL_COLUMNS:
        assert model_cats[c] <= set(df[c].astype(str)), c


@needs_artifacts
def test_form_covers_every_raw_field_and_defaulted_set_is_unimportant():
    missing = set(RAW_FEATURE_COLUMNS) - FORM_FIELDS - set(DERIVED_ON_FORM) - set(DOCUMENTED_DEFAULTED)
    assert not missing, f"raw fields neither collected, derived nor documented: {missing}"
    imp = pd.read_csv(PROC / "shap_importance_by_field.csv").set_index("source_field")["share_of_total"]
    assert imp.reindex(list(DOCUMENTED_DEFAULTED)).fillna(0).sum() < 0.05


# ============================== D2 =========================================
@needs_artifacts
@pytest.mark.parametrize("field,a,b,eng_col", [
    ("incident_severity", "Minor Damage", "Major Damage", "is_major_damage"),
    ("witnesses", 0, 3, "witnesses"),
    ("police_report_available", "YES", "NO", "police_report_available_NO"),
])
def test_one_variable_scenarios_stay_consistent(field, a, b, eng_col):
    svc = _svc()
    claims = pd.DataFrame([{**MISSING_COLUMN_DEFAULTS, field: a}, {**MISSING_COLUMN_DEFAULTS, field: b}])
    X_scaled, X_raw = svc._prepare(claims, return_raw=True)
    changed = [c for c in X_raw.columns if X_raw[c].iloc[0] != X_raw[c].iloc[1]]
    assert eng_col in changed
    from app.ml.explainer import build_source_map
    src = build_source_map(svc.feature_columns)
    assert {src.get(c, (c, None))[0] for c in changed} == {field}  # nothing else moved
    p = svc.model_pipeline.predict_proba(X_scaled)[:, 1]
    s1, s2 = svc.score_one(claims.iloc[0].to_dict()), svc.score_one(claims.iloc[1].to_dict())
    assert s1["fraud_probability"] == pytest.approx(p[0]) and s2["fraud_probability"] == pytest.approx(p[1])


# ============================== D3 =========================================
@needs_artifacts
def test_no_test_row_used_before_final_evaluation():
    ids = json.load(open(PROC / "dev_phase_row_ids.json"))
    test = set(ids["test_ids"])
    assert len(test) == 200 and len(ids["development_ids"]) == 800 and not test & set(ids["development_ids"])
    for step, used in ids["steps"].items():
        assert not test & set(used), f"{step} touched test rows"
        assert set(used) == set(ids["development_ids"]), step


@needs_artifacts
def test_champion_is_recomputed_identically_from_fold_results():
    from app.ml import model_selection as ms
    folds = pd.read_csv(PROC / "model_selection_folds.csv")
    cols = json.load(open(MODELS / "feature_columns.json"))
    again = ms.select_champion(folds, ms.build_candidates(cols))
    assert again["champion"] == json.load(open(MODELS / "metrics.json"))["primary_model"]


def test_no_hardcoded_champion_or_stale_numbers_in_code():
    # generate_thesis_numbers.py legitimately lists the thesis's OLD values.
    src = "\n".join(p.read_text() for p in (ROOT / "backend" / "app").rglob("*.py") if p.name != "generate_thesis_numbers.py")
    assert not re.search(r'champion\w*\s*=\s*"(random_forest|logistic_regression|xgboost)"', src)
    for stale in ("0.0086", "0.0166", "auto-approved"):
        assert stale not in src, stale


@needs_artifacts
def test_bands_and_threshold_come_from_artifacts():
    from app.ml.risk_policy import HIGH_RISK_EDGE, MEDIUM_RISK_EDGE
    pol = json.load(open(MODELS / "risk_policy.json"))
    assert (MEDIUM_RISK_EDGE, HIGH_RISK_EDGE) == (pol["medium_edge"], pol["high_edge"])
    assert _svc().operating_threshold == json.load(open(MODELS / "metrics.json"))["operating_threshold"]


def test_single_missing_component_is_derived_exactly_not_defaulted():
    """Regression for the 5,000-row demo bug: a blank property_claim used the
    training median and produced a 397% property share."""
    from app.ml.feature_engineering import apply_missing_defaults
    out = apply_missing_defaults(pd.DataFrame([{"total_claim_amount": 3297, "injury_claim": 0, "vehicle_claim": 3297}]))
    assert out["property_claim"].iloc[0] == 0
    X = engineer_features(pd.DataFrame([{"total_claim_amount": 12244, "injury_claim": 3270, "vehicle_claim": 5149}]))
    assert X["property_claim_pct"].iloc[0] <= 1.0
