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
from app.ml.feature_engineering import align_to_training_columns, engineer_features

MODEL_VERSION_FILE = MODELS_DIR / "metrics.json"


class FraudScoringService:
    _instance: "FraudScoringService | None" = None

    def __init__(self):
        self.rf_pipeline = joblib.load(MODELS_DIR / "random_forest_final.pkl")
        self.scaler = joblib.load(MODELS_DIR / "standard_scaler.pkl")
        self.zip3_lookup = pd.read_csv(MODELS_DIR / "zip3_lookup.csv")
        with open(MODELS_DIR / "feature_columns.json") as f:
            self.feature_columns: list[str] = json.load(f)
        with open(MODEL_VERSION_FILE) as f:
            metrics = json.load(f)
        self.operating_threshold = metrics["operating_threshold"]
        self.model_version = f"random_forest-{metrics['n_features']}f-{metrics['n_train']}train"

        rf_only = self.rf_pipeline.named_steps["rf"]
        self.explainer = ClaimExplainer(rf_only, self.feature_columns)

    @classmethod
    def instance(cls) -> "FraudScoringService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _prepare(self, claims: pd.DataFrame, return_raw: bool = False):
        X = engineer_features(claims, self.zip3_lookup)
        X = align_to_training_columns(X, self.feature_columns)
        X_scaled = pd.DataFrame(self.scaler.transform(X), columns=self.feature_columns, index=claims.index)
        return (X_scaled, X) if return_raw else X_scaled

    def score_batch(self, claims: pd.DataFrame) -> pd.DataFrame:
        X_scaled = self._prepare(claims)
        proba = self.rf_pipeline.predict_proba(X_scaled)[:, 1]
        flagged = proba >= self.operating_threshold
        grade = pd.cut(proba, bins=[-0.01, 0.3, 0.6, 1.01], labels=["Low", "Medium", "High"])
        return pd.DataFrame({
            "fraud_probability": proba, "risk_grade": grade.astype(str),
            "flagged": flagged, "operating_threshold": self.operating_threshold,
        }, index=claims.index)

    def score_one(self, claim: dict) -> dict:
        claim_df = pd.DataFrame([claim])
        X_scaled, X_raw = self._prepare(claim_df, return_raw=True)
        proba = float(self.rf_pipeline.predict_proba(X_scaled)[:, 1][0])
        reasons = self.explainer.top_reasons(X_scaled, k=3, X_row_raw=X_raw)
        grade = "High" if proba >= 0.6 else "Medium" if proba >= 0.3 else "Low"
        return {
            "fraud_probability": proba, "risk_grade": grade,
            "flagged": proba >= self.operating_threshold,
            "operating_threshold": self.operating_threshold,
            "top_reasons": reasons, "model_version": self.model_version,
        }
