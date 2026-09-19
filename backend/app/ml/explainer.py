"""
explainer.py — SHAP explanations for the shipped/compared models, plus a
plain-English reason-code generator so the dashboard and API can show
"why" a claim was flagged, not just a probability.

PB-05 (fixed): this used to unconditionally build `shap.TreeExplainer(model)`
regardless of what `model` actually was. That's correct for Random Forest
(the shipped champion) and XGBoost (a challenger), but this project also
trains and saves a Logistic Regression challenger
(`models/logistic_regression_final.pkl`) — and D4 explicitly allows a
future retrain's nested-CV evidence to pick LR as champion instead (see
docs/REBUILD_NOTES.md §11). TreeExplainer does not support linear models.
Reproduced directly:

    >>> shap.TreeExplainer(logistic_regression_estimator)
    shap.utils._exceptions.InvalidModelError: Model type not yet supported
    by TreeExplainer: <class 'sklearn.linear_model._logistic.LogisticRegression'>

This was never hit in practice because `ClaimExplainer` was only ever
instantiated with the RF estimator (inference.py, train.py) — a real bug
that was simply never exercised, not one that had been verified safe.
Fixed by picking the SHAP explainer class from the model's own type:
TreeExplainer for tree ensembles (Random Forest, XGBoost — no background
data needed, exact), LinearExplainer for linear models (Logistic
Regression — needs a background sample to estimate the feature
covariance/expectation). Falls back to the generic, model-agnostic
`shap.Explainer` on `predict_proba` for anything else, also using the
background sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier
    _TREE_MODEL_TYPES = (RandomForestClassifier, XGBClassifier)
except ImportError:  # pragma: no cover - xgboost is a hard dependency here, but stay defensive
    _TREE_MODEL_TYPES = (RandomForestClassifier,)

_LINEAR_MODEL_TYPES = (LogisticRegression,)

# PB-05: k=8 is the new default — 3 was too few to give an investigator a
# genuinely useful "why" for a borderline claim (a top-3 list frequently
# left out features that were themselves flagged elsewhere in the audit,
# e.g. is_highrisk_hobby / is_exec_occupation, purely because 2-3 slightly
# larger-magnitude features crowded them out).
DEFAULT_TOP_K = 8


class ClaimExplainer:
    def __init__(self, model, feature_columns: list[str], background_data: pd.DataFrame | None = None):
        """`model` is the bare estimator (e.g. `pipeline.named_steps["rf"]`)
        — never the full Pipeline(SMOTE -> classifier); SHAP needs to
        explain the classifier's actual decision function, not the
        resampling step. `background_data` is only required for
        non-tree models (a representative sample of SCALED features, the
        same space the model was fit on — a few dozen to a few hundred
        rows is enough); tree models ignore it."""
        self.model = model
        self.feature_columns = feature_columns

        if isinstance(model, _TREE_MODEL_TYPES):
            self.explainer = shap.TreeExplainer(model)
            self.explainer_kind = "tree"
        elif isinstance(model, _LINEAR_MODEL_TYPES):
            if background_data is None:
                raise ValueError(
                    f"ClaimExplainer for {type(model).__name__} requires background_data "
                    "(a sample of scaled features) — LinearExplainer cannot be built without one."
                )
            self.explainer = shap.LinearExplainer(model, background_data)
            self.explainer_kind = "linear"
        else:
            if background_data is None:
                raise ValueError(
                    f"ClaimExplainer for unrecognized model type {type(model).__name__} requires "
                    "background_data — falling back to the generic, sampling-based shap.Explainer."
                )
            self.explainer = shap.Explainer(model.predict_proba, background_data)
            self.explainer_kind = "generic"

    def shap_values_for(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.explainer.shap_values(X) if hasattr(self.explainer, "shap_values") else self.explainer(X).values
        if isinstance(raw, list):  # older SHAP returns [class0, class1] for classifiers
            return np.asarray(raw[1])
        raw = np.asarray(raw)
        if raw.ndim == 3:  # newer SHAP returns (n_rows, n_features, n_classes)
            return raw[:, :, 1]
        return raw  # tree/linear binary explainers already return (n_rows, n_features)

    def global_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        sv = self.shap_values_for(X)
        mean_abs = np.abs(sv).mean(axis=0)
        return (
            pd.DataFrame({"feature": self.feature_columns, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

    def top_reasons(self, X_row_scaled: pd.DataFrame, k: int = DEFAULT_TOP_K, X_row_raw: pd.DataFrame | None = None) -> list[dict]:
        """`X_row_scaled` drives the actual SHAP computation (matches what
        the model was fit on); `X_row_raw` (pre-scaling engineered values),
        when given, is what gets shown to a human — a raw one-hot 1/0 or an
        unscaled ratio reads far better in a reason sentence than that same
        value's z-score.

        PB-05: each reason is now a genuinely plain-language dict, not just
        a feature name + a raw signed float. `direction` and `impact` are
        words ("increased"/"decreased", "strongly"/"moderately"/"slightly"
        — impact is relative to the STRONGEST driver for this specific
        claim, since raw SHAP magnitude has no fixed, human-meaningful
        scale on its own), `display_name`/`display_value` are the
        human-readable feature name and formatted value used to build
        `sentence`, and `rank` is the reason's 1-based position so a caller
        doesn't have to infer order from list position."""
        sv = self.shap_values_for(X_row_scaled)[0]
        display_row = X_row_raw if X_row_raw is not None else X_row_scaled
        order = np.argsort(-np.abs(sv))[:k]
        max_abs = float(np.abs(sv).max()) if len(sv) else 0.0
        reasons = []
        for rank, idx in enumerate(order, start=1):
            feat = self.feature_columns[idx]
            # PB-05: decide "is this a yes/no flag" from the COLUMN's dtype,
            # not from the value itself — a genuine numeric feature (e.g.
            # witnesses=1, or a ratio that happens to equal 0.0) must never
            # be mislabeled "yes"/"no" just because its value looks
            # boolean-ish. One-hot columns are bool dtype when the row's
            # own category produced them, or int dtype (fill_value=0) when
            # align_to_training_columns() padded in a category this row
            # doesn't belong to (see feature_engineering.py) — both read
            # correctly as a flag under is_bool_dtype-or-name check below.
            col_is_flag = pd.api.types.is_bool_dtype(display_row.dtypes.iloc[idx]) or feat in ("is_highrisk_hobby", "is_exec_occupation", "is_new_customer", "is_no_witness", "is_major_damage")
            raw_val = display_row.iloc[0, idx]
            if hasattr(raw_val, "item"):  # numpy scalar -> plain python
                raw_val = raw_val.item()
            direction = "increased" if sv[idx] > 0 else "decreased"
            impact = _impact_label(abs(float(sv[idx])), max_abs)
            display_value = _format_value(raw_val, is_flag=col_is_flag)
            reasons.append({
                "rank": rank,
                "feature": feat,
                "display_name": _humanize(feat),
                "value": bool(raw_val) if col_is_flag else (float(raw_val) if isinstance(raw_val, (int, float, np.number)) else str(raw_val)),
                "shap_value": float(sv[idx]),
                "direction": direction,
                "impact": impact,
                "sentence": f"{_humanize(feat)} ({display_value}) {impact} {direction} the fraud risk score.",
            })
        return reasons


def _impact_label(abs_shap: float, max_abs_shap_in_row: float) -> str:
    if max_abs_shap_in_row <= 0:
        return "slightly"
    ratio = abs_shap / max_abs_shap_in_row
    if ratio >= 0.66:
        return "strongly"
    if ratio >= 0.33:
        return "moderately"
    return "slightly"


def _format_value(val, is_flag: bool = False) -> str:
    """Human-readable rendering of a single raw/engineered feature value
    for use inside a reason sentence — yes/no for flags (decided by the
    caller from column dtype/name, never guessed from the value alone —
    see top_reasons()), plain ints, 2-decimal floats, everything else
    as-is."""
    if is_flag or isinstance(val, (bool, np.bool_)):
        return "yes" if val else "no"
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    if isinstance(val, (float, np.floating)):
        fv = float(val)
        if fv == int(fv):
            return str(int(fv))
        return f"{fv:.2f}"
    return str(val)


def _humanize(feature: str) -> str:
    return feature.replace("_", " ").replace("-", " ").strip()
