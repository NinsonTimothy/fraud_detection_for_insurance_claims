"""
tests/test_prototype.py — Automated validation for the fraud risk scoring prototype.

Run with:
    cd prototype
    pytest tests/ -v

These tests import directly from scoring.py (no Streamlit involved), so they run headlessly
and can be re-run any time app.py or scoring.py changes, without needing a browser.

Tests that need the trained model / SHAP explainer / X_train.csv on disk are skipped
automatically (with a clear reason) if those files aren't present — e.g. if you're running
this in CI or a fresh clone before the model artifacts have been generated.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import scoring  # noqa: E402


ARTIFACTS_MISSING = scoring.artifacts_available()
requires_artifacts = pytest.mark.skipif(
    bool(ARTIFACTS_MISSING),
    reason=f"Model artifacts not found: {ARTIFACTS_MISSING}",
)


# --------------------------------------------------------------------------------------
# Pure logic tests — no model artifacts needed at all
# --------------------------------------------------------------------------------------
class TestRiskBanding:
    def test_low_band_boundaries(self):
        assert scoring.assign_band(0) == "Low"
        assert scoring.assign_band(29) == "Low"

    def test_medium_band_boundaries(self):
        assert scoring.assign_band(30) == "Medium"
        assert scoring.assign_band(59) == "Medium"

    def test_high_band_boundaries(self):
        assert scoring.assign_band(60) == "High"
        assert scoring.assign_band(100) == "High"

    def test_fraud_risk_score_rounds_correctly(self):
        result = scoring.fraud_risk_score(0.554)
        assert result["risk_score"] == 55
        assert result["risk_band"] == "Medium"

    def test_fraud_risk_score_high_probability(self):
        result = scoring.fraud_risk_score(0.91)
        assert result["risk_score"] == 91
        assert result["risk_band"] == "High"

    def test_fraud_risk_score_zero_probability(self):
        result = scoring.fraud_risk_score(0.0)
        assert result["risk_score"] == 0
        assert result["risk_band"] == "Low"


class TestFeatureBuilder:
    """These only need a synthetic column schema, not the real trained model."""

    @pytest.fixture
    def template_columns(self):
        # A minimal but representative schema covering every column build_feature_row touches
        cols = [
            "months_as_customer", "age", "policy_annual_premium", "policy_deductable",
            "incident_hour_of_the_day", "number_of_vehicles_involved", "bodily_injuries",
            "witnesses", "capital_gains", "capital_loss", "injury_claim", "property_claim",
            "vehicle_claim", "umbrella_limit", "claim_to_premium_ratio", "vehicle_claim_pct",
            "injury_claim_pct", "car_age", "policy_age_at_incident_days", "csl_per_person",
            "csl_total", "is_major_damage", "is_no_witness", "is_highrisk_hobby",
            "is_exec_occupation", "is_new_customer", "is_weekend", "incident_severity",
            "hour_bucket_Night", "hour_bucket_Rush_Hour",
            "zip3_risk_tier_low_risk", "zip3_risk_tier_elevated", "zip3_risk_tier_high_risk",
            "incident_type_Single Vehicle Collision", "collision_type_Front Collision",
            "authorities_contacted_Police", "incident_state_NY", "insured_hobbies_chess",
            "auto_region_USA", "policy_state_OH",
        ]
        return pd.Index(cols)

    @pytest.fixture
    def sample_raw_claim(self):
        return dict(
            age=41, months_as_customer=96, policy_annual_premium=1100.0, policy_deductable=1000,
            umbrella_limit=0, capital_gains=0, capital_loss=0, csl_per_person=100000, csl_total=300000,
            incident_type="Single Vehicle Collision", collision_type="Front Collision",
            incident_severity="Major Damage", authorities_contacted="Police", incident_state="NY",
            incident_hour_of_the_day=2, number_of_vehicles_involved=1, bodily_injuries=0, witnesses=0,
            injury_claim=15000, property_claim=8000, vehicle_claim=42000,
            insured_hobbies="chess", insured_occupation="exec-managerial", auto_region="USA",
            auto_year=2016, policy_state="OH",
            policy_bind_date=pd.Timestamp("2016-01-15"), incident_date=pd.Timestamp("2024-03-02"),
        )

    def test_returns_correct_shape(self, sample_raw_claim, template_columns):
        row, _ = scoring.build_feature_row(sample_raw_claim, template_columns)
        assert row.shape == (1, len(template_columns))
        assert list(row.columns) == list(template_columns)

    def test_direct_numeric_fields_pass_through(self, sample_raw_claim, template_columns):
        row, _ = scoring.build_feature_row(sample_raw_claim, template_columns)
        assert row.at[0, "age"] == 41
        assert row.at[0, "vehicle_claim"] == 42000

    def test_is_major_damage_flag(self, sample_raw_claim, template_columns):
        row, _ = scoring.build_feature_row(sample_raw_claim, template_columns)
        assert row.at[0, "is_major_damage"] == 1

    def test_is_no_witness_flag(self, sample_raw_claim, template_columns):
        row, _ = scoring.build_feature_row(sample_raw_claim, template_columns)
        assert row.at[0, "is_no_witness"] == 1

    def test_high_risk_hobby_flag(self, sample_raw_claim, template_columns):
        row, _ = scoring.build_feature_row(sample_raw_claim, template_columns)
        assert row.at[0, "is_highrisk_hobby"] == 1

    def test_exec_occupation_flag(self, sample_raw_claim, template_columns):
        row, _ = scoring.build_feature_row(sample_raw_claim, template_columns)
        assert row.at[0, "is_exec_occupation"] == 1

    def test_hour_bucket_night_for_2am(self, sample_raw_claim, template_columns):
        row, unmatched = scoring.build_feature_row(sample_raw_claim, template_columns)
        assert row.at[0, "hour_bucket_Night"] == 1
        assert "hour_bucket_Night" not in unmatched

    def test_one_hot_categorical_matching(self, sample_raw_claim, template_columns):
        row, unmatched = scoring.build_feature_row(sample_raw_claim, template_columns)
        assert row.at[0, "incident_type_Single Vehicle Collision"] == 1
        assert row.at[0, "authorities_contacted_Police"] == 1
        assert row.at[0, "incident_state_NY"] == 1

    def test_unmatched_category_is_flagged_not_silently_dropped(self, sample_raw_claim, template_columns):
        sample_raw_claim["incident_state"] = "ZZ"  # not in the synthetic schema
        row, unmatched = scoring.build_feature_row(sample_raw_claim, template_columns)
        assert "incident_state_ZZ" in unmatched

    def test_claim_to_premium_ratio_calculation(self, sample_raw_claim, template_columns):
        row, _ = scoring.build_feature_row(sample_raw_claim, template_columns)
        expected_total = 15000 + 8000 + 42000
        expected_ratio = expected_total / 1100.0
        assert row.at[0, "claim_to_premium_ratio"] == pytest.approx(expected_ratio)

    def test_zero_premium_does_not_divide_by_zero(self, sample_raw_claim, template_columns):
        sample_raw_claim["policy_annual_premium"] = 0
        row, _ = scoring.build_feature_row(sample_raw_claim, template_columns)
        assert row.at[0, "claim_to_premium_ratio"] == 0

    def test_severity_ordinal_mapping(self, sample_raw_claim, template_columns):
        row, _ = scoring.build_feature_row(sample_raw_claim, template_columns)
        assert row.at[0, "incident_severity"] == 2  # Major Damage -> 2

    def test_car_age_uses_incident_date_year_not_a_separate_field(self, sample_raw_claim, template_columns):
        row, _ = scoring.build_feature_row(sample_raw_claim, template_columns)
        # incident_date is 2024-03-02, auto_year is 2016
        assert row.at[0, "car_age"] == 8

    def test_is_new_customer_flag(self, sample_raw_claim, template_columns):
        sample_raw_claim["policy_bind_date"] = pd.Timestamp("2024-01-01")
        sample_raw_claim["incident_date"] = pd.Timestamp("2024-03-02")  # 61 days later
        row, _ = scoring.build_feature_row(sample_raw_claim, template_columns)
        assert row.at[0, "is_new_customer"] == 1

    def test_is_weekend_flag(self, sample_raw_claim, template_columns):
        sample_raw_claim["incident_date"] = pd.Timestamp("2024-03-02")  # a Saturday
        row, _ = scoring.build_feature_row(sample_raw_claim, template_columns)
        assert row.at[0, "is_weekend"] == 1

    def test_claim_pct_uses_plus_one_smoothing(self, sample_raw_claim, template_columns):
        row, _ = scoring.build_feature_row(sample_raw_claim, template_columns)
        total = 15000 + 8000 + 42000
        assert row.at[0, "vehicle_claim_pct"] == pytest.approx(42000 / (total + 1))

    def test_rush_hour_bucket(self, sample_raw_claim, template_columns):
        sample_raw_claim["incident_hour_of_the_day"] = 8  # falls in 07-09 rush hour window
        row, unmatched = scoring.build_feature_row(sample_raw_claim, template_columns)
        assert row.at[0, "hour_bucket_Rush_Hour"] == 1

    def test_policy_state_one_hot(self, sample_raw_claim, template_columns):
        row, _ = scoring.build_feature_row(sample_raw_claim, template_columns)
        assert row.at[0, "policy_state_OH"] == 1


# --------------------------------------------------------------------------------------
# Integration tests — need the real trained model, SHAP explainer, and X_train schema
# --------------------------------------------------------------------------------------
@requires_artifacts
class TestModelIntegration:
    @pytest.fixture(scope="class")
    def artifacts(self):
        return scoring.load_artifacts()

    def test_artifacts_load_without_error(self, artifacts):
        model, explainer, template_columns = artifacts
        assert model is not None
        assert explainer is not None
        assert len(template_columns) > 0

    def test_model_predict_proba_returns_valid_range(self, artifacts):
        model, explainer, template_columns = artifacts
        row = pd.DataFrame(0, index=[0], columns=template_columns)
        prob = model.predict_proba(row)[0, 1]
        assert 0.0 <= prob <= 1.0

    def test_score_claim_end_to_end(self, artifacts):
        model, explainer, template_columns = artifacts
        raw = dict(
            age=41, months_as_customer=96, policy_annual_premium=1100.0, policy_deductable=1000,
            umbrella_limit=0, capital_gains=0, capital_loss=0, csl_per_person=100000, csl_total=300000,
            incident_type="Single Vehicle Collision", collision_type="Not Applicable",
            incident_severity="Major Damage", authorities_contacted="None", incident_state="WV",
            incident_hour_of_the_day=2, number_of_vehicles_involved=1, bodily_injuries=0, witnesses=0,
            injury_claim=15000, property_claim=8000, vehicle_claim=42000,
            insured_hobbies="chess", insured_occupation="exec-managerial", auto_region="Other",
            auto_year=2016, policy_state="OH",
            policy_bind_date=pd.Timestamp("2016-01-15"), incident_date=pd.Timestamp("2024-03-02"),
        )
        row, _ = scoring.build_feature_row(raw, template_columns)
        result = scoring.score_claim(model, explainer, row)

        assert 0.0 <= result["fraud_probability"] <= 1.0
        assert 0 <= result["risk_score"] <= 100
        assert result["risk_band"] in ("Low", "Medium", "High")
        assert len(result["top_shap_factors"]) == 8


@requires_artifacts
class TestKnownCasesFromRealTestSet:
    """
    Mirrors Track A from 06_prototype.ipynb: pulls real, already-encoded rows from
    X_test.csv / y_test.csv and checks the pipeline reproduces expected behaviour.
    Skipped automatically if those files aren't present alongside the model artifacts.
    """

    @pytest.fixture(scope="class")
    def real_test_data(self):
        x_test_path = os.path.join(scoring.DATA_DIR, "X_test.csv")
        y_test_path = os.path.join(scoring.DATA_DIR, "y_test.csv")
        if not (os.path.exists(x_test_path) and os.path.exists(y_test_path)):
            pytest.skip("X_test.csv / y_test.csv not found alongside model artifacts")
        X_test = pd.read_csv(x_test_path).rename(
            columns={"capital-gains": "capital_gains", "capital-loss": "capital_loss"}
        )
        y_test = pd.read_csv(y_test_path).squeeze()
        return X_test, y_test

    def test_known_fraud_case_lands_in_medium_or_high_band(self, real_test_data):
        model, explainer, _ = scoring.load_artifacts()
        X_test, y_test = real_test_data
        y_prob_all = model.predict_proba(X_test)[:, 1]

        fraud_idx = y_test[y_test == 1].index
        best_fraud_pos = pd.Series(y_prob_all, index=y_test.index).loc[fraud_idx].idxmax()
        pos = X_test.index.get_loc(best_fraud_pos)

        result = scoring.score_claim(model, explainer, X_test.iloc[[pos]])
        assert result["risk_band"] in ("Medium", "High")

    def test_known_nonfraud_case_lands_in_low_band(self, real_test_data):
        model, explainer, _ = scoring.load_artifacts()
        X_test, y_test = real_test_data
        y_prob_all = model.predict_proba(X_test)[:, 1]

        nonfraud_idx = y_test[y_test == 0].index
        best_nonfraud_pos = pd.Series(y_prob_all, index=y_test.index).loc[nonfraud_idx].idxmin()
        pos = X_test.index.get_loc(best_nonfraud_pos)

        result = scoring.score_claim(model, explainer, X_test.iloc[[pos]])
        assert result["risk_band"] == "Low"
