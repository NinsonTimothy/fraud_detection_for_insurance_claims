"""
inference.py — FraudScoringService: loads the CHAMPION model (whichever model
the pre-declared selection rule picked — see model_selection.py), the scaler,
the derived risk bands and the explainer ONCE, then scores in-process. The
FastAPI backend and the Streamlit dashboard share this class, so they can
never disagree about a score.

The champion artifact may be a bare estimator or a CalibratedClassifierCV
(ensemble=False) wrapper (B4). SHAP always explains the single underlying
estimator; the calibrator only re-maps its score monotonically, so the
ranking — and the direction of every explanation — is unchanged.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from app.core.config import MODELS_DIR
from app.ml.explainer import ClaimExplainer, split_reasons
from app.ml.feature_engineering import RAW_FEATURE_COLUMNS, align_to_training_columns, apply_missing_defaults, engineer_features
from app.ml.risk_policy import DECISION_SUPPORT_NOTICE, POLICY_FILE, grade_for, grade_for_array, is_flagged, recommended_action

MODEL_VERSION_FILE = MODELS_DIR / "metrics.json"
DERIVED_FIELDS = ("claim_to_premium_ratio", "vehicle_claim_pct", "injury_claim_pct", "property_claim_pct",
                  "policy_age_at_incident_days", "vehicle_age_at_incident", "is_new_customer", "is_major_damage")


def underlying_estimator(model):
    """The single fitted estimator SHAP should explain."""
    if isinstance(model, CalibratedClassifierCV):
        model = model.calibrated_classifiers_[0].estimator
    if hasattr(model, "steps"):
        model = model.steps[-1][1]
    return model


class FraudScoringService:
    _instance: "FraudScoringService | None" = None

    def __init__(self):
        champion_path = MODELS_DIR / "champion_model.pkl"
        if not champion_path.exists():
            champion_path = MODELS_DIR / "random_forest_final.pkl"
        self.model_pipeline = joblib.load(champion_path)
        self.scaler = joblib.load(MODELS_DIR / "standard_scaler.pkl")
        with open(MODELS_DIR / "feature_columns.json") as f:
            self.feature_columns: list[str] = json.load(f)
        with open(MODEL_VERSION_FILE) as f:
            metrics = json.load(f)
        self.operating_threshold = float(metrics["operating_threshold"])
        self.model_name = metrics.get("primary_model", "random_forest")
        self.calibration = metrics.get("calibration", {}).get("chosen", "none")
        self.model_version = f"{self.model_name}-{metrics['n_features']}f-{metrics['n_train']}dev"
        try:
            with open(POLICY_FILE) as f:
                self.band_edges = json.load(f)
        except OSError:
            self.band_edges = None
        bg_path = MODELS_DIR / "shap_background.csv"
        background = pd.read_csv(bg_path)[self.feature_columns] if bg_path.exists() else None
        self.explainer = ClaimExplainer(underlying_estimator(self.model_pipeline), self.feature_columns,
                                        background_data=background)

    @classmethod
    def instance(cls) -> "FraudScoringService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def rf_pipeline(self):  # backward-compatible alias: it is the CHAMPION
        return self.model_pipeline

    def _prepare(self, claims: pd.DataFrame, return_raw: bool = False):
        X = align_to_training_columns(engineer_features(claims), self.feature_columns)
        X_scaled = pd.DataFrame(self.scaler.transform(X), columns=self.feature_columns, index=claims.index)
        return (X_scaled, X) if return_raw else X_scaled

    def _proba(self, X_scaled: pd.DataFrame) -> np.ndarray:
        return self.model_pipeline.predict_proba(X_scaled)[:, 1]

    def derived_values(self, claims: pd.DataFrame) -> pd.DataFrame:
        X = engineer_features(claims)
        return X[[c for c in DERIVED_FIELDS if c in X.columns]]

    def score_batch(self, claims: pd.DataFrame, with_field_shap: bool = False):
        X_scaled, X_raw = self._prepare(claims, return_raw=True)
        proba = self._proba(X_scaled)
        grade = grade_for_array(proba, self.band_edges)
        reasons = self.explainer.top_reasons_batch(X_scaled, X_raw=X_raw)
        out = pd.DataFrame({
            "fraud_probability": proba, "risk_grade": grade,
            "flagged": proba >= self.operating_threshold, "operating_threshold": self.operating_threshold,
            "recommended_action": [recommended_action(g) for g in grade], "top_reasons": reasons,
        }, index=claims.index)
        if with_field_shap:
            return out, self.explainer.grouped_shap_matrix(X_scaled)
        return out

    def score_one(self, claim: dict) -> dict:
        claim_df = pd.DataFrame([claim])
        X_scaled, X_raw = self._prepare(claim_df, return_raw=True)
        proba = float(self._proba(X_scaled)[0])
        reasons = self.explainer.top_reasons(X_scaled, k=len(self.feature_columns), X_row_raw=X_raw)
        up, down = split_reasons(reasons)
        grade = grade_for(proba, self.band_edges)
        derived = {k: float(v) for k, v in self.derived_values(apply_missing_defaults(claim_df)).iloc[0].items()}
        return {
            "fraud_probability": proba, "risk_grade": grade,
            "flagged": is_flagged(proba, self.operating_threshold),
            "operating_threshold": self.operating_threshold,
            "recommended_action": recommended_action(grade),
            "defaulted_fields": sorted(set(RAW_FEATURE_COLUMNS) - set(claim.keys())),
            "top_reasons": reasons[:8], "factors_increasing_risk": up, "factors_reducing_risk": down,
            "derived_values": derived, "model_version": self.model_version,
            "decision_support_notice": DECISION_SUPPORT_NOTICE,
        }
