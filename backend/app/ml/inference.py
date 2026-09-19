"""
inference.py — FraudScoringService: loads every model/scaler/explainer
artifact ONCE at process startup, then scores claims in-process. Both the
FastAPI backend and the Streamlit dashboard import this same class (the
dashboard calls it in-process, not over HTTP — same pattern as the sibling
MoMo Guard project, so the API and the dashboard can never disagree about a
score).
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from app.core.config import MODELS_DIR
from app.ml.explainer import ClaimExplainer
from app.ml.feature_engineering import RAW_FEATURE_COLUMNS, align_to_training_columns, engineer_features
from app.ml.risk_policy import grade_for, grade_for_array, is_flagged, recommended_action

MODEL_VERSION_FILE = MODELS_DIR / "metrics.json"


class FraudScoringService:
    _instance: "FraudScoringService | None" = None

    def __init__(self):
        self.rf_pipeline = joblib.load(MODELS_DIR / "random_forest_final.pkl")
        self.scaler = joblib.load(MODELS_DIR / "standard_scaler.pkl")
        with open(MODELS_DIR / "feature_columns.json") as f:
            self.feature_columns: list[str] = json.load(f)
        with open(MODEL_VERSION_FILE) as f:
            metrics = json.load(f)
        self.operating_threshold = metrics["operating_threshold"]
        self.model_version = f"random_forest-{metrics['n_features']}f-{metrics['n_train']}train"

        rf_only = self.rf_pipeline.named_steps["rf"]
        # PB-05: ClaimExplainer now picks the correct SHAP explainer from
        # the model's own type (TreeExplainer here, since RF is a tree
        # ensemble — no background_data needed). If a future retrain's
        # nested-CV evidence ever swaps the shipped champion to Logistic
        # Regression (D4 allows this), ClaimExplainer would need a
        # background_data sample passed here too — see explainer.py.
        self.explainer = ClaimExplainer(rf_only, self.feature_columns)

    @classmethod
    def instance(cls) -> "FraudScoringService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _prepare(self, claims: pd.DataFrame, return_raw: bool = False):
        X = engineer_features(claims)
        X = align_to_training_columns(X, self.feature_columns)
        X_scaled = pd.DataFrame(self.scaler.transform(X), columns=self.feature_columns, index=claims.index)
        return (X_scaled, X) if return_raw else X_scaled

    def score_batch(self, claims: pd.DataFrame) -> pd.DataFrame:
        X_scaled = self._prepare(claims)
        proba = self.rf_pipeline.predict_proba(X_scaled)[:, 1]
        # PB-04: grade_for_array() and score_one()'s grade_for() are the
        # SAME function's vectorized/scalar forms (risk_policy.py) — they
        # can no longer disagree at the band edges the way the old
        # pd.cut()-vs-hand-written-if/elif pair did.
        grade = grade_for_array(proba)
        flagged = is_flagged(proba, self.operating_threshold)
        return pd.DataFrame({
            "fraud_probability": proba, "risk_grade": grade,
            "flagged": flagged, "operating_threshold": self.operating_threshold,
            "recommended_action": [recommended_action(g) for g in grade],
        }, index=claims.index)

    def score_one(self, claim: dict) -> dict:
        claim_df = pd.DataFrame([claim])
        X_scaled, X_raw = self._prepare(claim_df, return_raw=True)
        proba = float(self.rf_pipeline.predict_proba(X_scaled)[:, 1][0])
        # PB-05: k defaults to explainer.DEFAULT_TOP_K (8) — was hardcoded
        # to 3 here, too few for a genuinely useful "why" (see explainer.py
        # module docstring / DEFAULT_TOP_K comment).
        reasons = self.explainer.top_reasons(X_scaled, X_row_raw=X_raw)
        grade = grade_for(proba)
        # PB-06: report which raw fields this claim did NOT supply (and
        # therefore fell back to feature_engineering.MISSING_COLUMN_DEFAULTS
        # for) — so a caller can tell a real-data score from a
        # mostly-defaulted one instead of the two looking identical.
        defaulted_fields = sorted(set(RAW_FEATURE_COLUMNS) - set(claim.keys()))
        return {
            "fraud_probability": proba, "risk_grade": grade,
            "flagged": is_flagged(proba, self.operating_threshold),
            "operating_threshold": self.operating_threshold,
            "recommended_action": recommended_action(grade),
            "defaulted_fields": defaulted_fields,
            "top_reasons": reasons, "model_version": self.model_version,
        }
