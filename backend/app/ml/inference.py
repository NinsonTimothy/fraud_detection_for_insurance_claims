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
from app.ml.risk_policy import DECISION_SUPPORT_NOTICE, grade_for, grade_for_array, is_flagged, recommended_action

MODEL_VERSION_FILE = MODELS_DIR / "metrics.json"


class FraudScoringService:
    _instance: "FraudScoringService | None" = None

    def __init__(self):
        # MS-01: the shipped model is whichever candidate won the leak-free
        # selection in train.py (models/champion_model.pkl) — not hardcoded
        # to Random Forest any more. Falls back to the legacy RF artifact
        # only if an older models/ directory has no champion file.
        champion_path = MODELS_DIR / "champion_model.pkl"
        if not champion_path.exists():
            champion_path = MODELS_DIR / "random_forest_final.pkl"
        self.model_pipeline = joblib.load(champion_path)
        self.scaler = joblib.load(MODELS_DIR / "standard_scaler.pkl")
        with open(MODELS_DIR / "feature_columns.json") as f:
            self.feature_columns: list[str] = json.load(f)
        with open(MODEL_VERSION_FILE) as f:
            metrics = json.load(f)
        self.operating_threshold = metrics["operating_threshold"]
        self.model_name = metrics.get("primary_model", "random_forest")
        self.model_version = f"{self.model_name}-{metrics['n_features']}f-{metrics['n_train']}train"

        final_estimator = self.model_pipeline.steps[-1][1] if hasattr(self.model_pipeline, "steps") else self.model_pipeline
        # Tree models need no background data; a linear champion (Logistic
        # Regression) needs the scaled training sample train.py saves.
        background_path = MODELS_DIR / "shap_background.csv"
        background = pd.read_csv(background_path)[self.feature_columns] if background_path.exists() else None
        self.explainer = ClaimExplainer(final_estimator, self.feature_columns, background_data=background)

    @property
    def rf_pipeline(self):
        """Backward-compatible alias (older code/tests called the shipped
        model `rf_pipeline`). It is the CHAMPION pipeline, whatever it is."""
        return self.model_pipeline

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
        X_scaled, X_raw = self._prepare(claims, return_raw=True)
        proba = self.model_pipeline.predict_proba(X_scaled)[:, 1]
        # PB-04: grade_for_array() and score_one()'s grade_for() are the
        # SAME function's vectorized/scalar forms (risk_policy.py) — they
        # can no longer disagree at the band edges the way the old
        # pd.cut()-vs-hand-written-if/elif pair did.
        grade = grade_for_array(proba)
        flagged = is_flagged(proba, self.operating_threshold)
        # PB-18: batch scoring used to skip SHAP entirely (no top_reasons
        # column at all), so every claim persisted through /score/batch or
        # the dashboard's Batch review page had no "why" — unlike a
        # single-claim score, which always got one. top_reasons_batch()
        # (explainer.py) computes SHAP for the whole batch in ONE call
        # (already vectorized across rows), not N single-row calls, so
        # this is the batch equivalent of score_one()'s explanation, not a
        # slower re-implementation of it.
        reasons = self.explainer.top_reasons_batch(X_scaled, X_raw=X_raw)
        return pd.DataFrame({
            "fraud_probability": proba, "risk_grade": grade,
            "flagged": flagged, "operating_threshold": self.operating_threshold,
            "recommended_action": [recommended_action(g) for g in grade],
            "top_reasons": reasons,
        }, index=claims.index)

    def score_one(self, claim: dict) -> dict:
        claim_df = pd.DataFrame([claim])
        X_scaled, X_raw = self._prepare(claim_df, return_raw=True)
        proba = float(self.model_pipeline.predict_proba(X_scaled)[:, 1][0])
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
            "decision_support_notice": DECISION_SUPPORT_NOTICE,
        }
