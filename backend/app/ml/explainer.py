"""
explainer.py — SHAP TreeExplainer wrapping the shipped model's underlying
tree ensemble, plus a plain-English reason-code generator so the dashboard
and API can show "why" a claim was flagged, not just a probability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap


class ClaimExplainer:
    def __init__(self, tree_model, feature_columns: list[str]):
        # For a Pipeline(SMOTE -> RandomForestClassifier), explain the
        # final estimator directly on already-engineered features — SHAP's
        # TreeExplainer needs the raw tree model, not the resampling step.
        self.explainer = shap.TreeExplainer(tree_model)
        self.feature_columns = feature_columns

    def shap_values_for(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.explainer.shap_values(X)
        if isinstance(raw, list):  # older SHAP returns [class0, class1] for classifiers
            return raw[1]
        if raw.ndim == 3:  # newer SHAP returns (n_rows, n_features, n_classes)
            return raw[:, :, 1]
        return raw

    def global_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        sv = self.shap_values_for(X)
        mean_abs = np.abs(sv).mean(axis=0)
        return (
            pd.DataFrame({"feature": self.feature_columns, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

    def top_reasons(self, X_row_scaled: pd.DataFrame, k: int = 3, X_row_raw: pd.DataFrame | None = None) -> list[dict]:
        """`X_row_scaled` drives the actual SHAP computation (matches what
        the model was fit on); `X_row_raw` (pre-scaling engineered values),
        when given, is what gets shown to a human — a raw one-hot 1/0 or an
        unscaled ratio reads far better in a reason sentence than that same
        value's z-score."""
        sv = self.shap_values_for(X_row_scaled)[0]
        display_row = X_row_raw if X_row_raw is not None else X_row_scaled
        order = np.argsort(-np.abs(sv))[:k]
        reasons = []
        for idx in order:
            feat = self.feature_columns[idx]
            val = display_row.iloc[0, idx]
            direction = "raised" if sv[idx] > 0 else "lowered"
            reasons.append({
                "feature": feat, "value": float(val) if isinstance(val, (int, float, np.number)) else str(val),
                "shap_value": float(sv[idx]),
                "sentence": f"{_humanize(feat)} {direction} the fraud risk score ({sv[idx]:+.3f}).",
            })
        return reasons


def _humanize(feature: str) -> str:
    return feature.replace("_", " ").replace("-", " ").strip()
