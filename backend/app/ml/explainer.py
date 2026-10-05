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
        self._source_of = build_source_map(feature_columns)

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
        return self._reasons_for_row(sv, display_row.iloc[0], display_row.dtypes, k)

    def top_reasons_batch(self, X_scaled: pd.DataFrame, k: int = DEFAULT_TOP_K, X_raw: pd.DataFrame | None = None) -> list[list[dict]]:
        """Batch counterpart to `top_reasons()` (PB-18) — ONE SHAP call over
        the whole matrix instead of N single-row calls. `shap_values_for()`
        is already vectorized across rows for every explainer kind this
        class builds (TreeExplainer/LinearExplainer/generic `Explainer`),
        so the SHAP computation itself costs the same total work either
        way; what this avoids is N-1 *extra* Python/SHAP call-overhead
        round-trips, which is what made per-row explanation look too
        expensive to bother with for batch scoring before this ticket
        (`score_batch()` shipped with no `top_reasons` at all). Returns one
        reasons list per row, same order as `X_scaled`."""
        sv_matrix = self.shap_values_for(X_scaled)
        display = X_raw if X_raw is not None else X_scaled
        dtypes = display.dtypes  # same for every row — compute once, not once per row
        return [self._reasons_for_row(sv_matrix[i], display.iloc[i], dtypes, k) for i in range(len(display))]

    def _reasons_for_row(self, sv: np.ndarray, raw_values: pd.Series, dtypes: pd.Series, k: int) -> list[dict]:
        """EX-01 (pre-defence fix): one reason per ORIGINAL claim field.

        The bug: categorical fields are one-hot encoded, so "police report
        available" is really two model columns, `police_report_available_NO`
        and `police_report_available_YES`. Each column used to become its own
        reason, labelled with the column's CATEGORY name. For a claim where
        the analyst selected "NO", the `_YES` column (value 0, meaning "not
        YES") still produced a reason reading "police report available YES
        (0) ... decreased the fraud risk" — which every reviewer read as
        "the claim HAS a police report". The model was right; the label was
        wrong.

        The fix: SHAP values are additive, so the contributions of all the
        dummy columns that came from one categorical field are summed into
        ONE reason, and that reason is labelled with the category this claim
        actually has (the dummy whose value is 1). "Police report available:
        NO" is now the only thing an investigator can see. Numeric/flag
        features are unaffected apart from friendlier labels."""
        groups: dict[str, dict] = {}
        for idx, feat in enumerate(self.feature_columns):
            source, category = self._source_of.get(feat, (feat, None))
            g = groups.setdefault(source, {"shap": 0.0, "category": None, "idx": idx, "is_cat": category is not None})
            g["shap"] += float(sv[idx])
            if category is not None:
                val = raw_values.iloc[idx]
                if hasattr(val, "item"):
                    val = val.item()
                if bool(val):
                    g["category"] = category

        ordered = sorted(groups.items(), key=lambda kv: -abs(kv[1]["shap"]))[:k]
        max_abs = max((abs(g["shap"]) for _, g in ordered), default=0.0)
        reasons = []
        for rank, (source, g) in enumerate(ordered, start=1):
            label = FEATURE_LABELS.get(source, _humanize(source))
            if g["is_cat"]:
                value = g["category"] if g["category"] is not None else "other / not seen in training"
                display_value = value
                value_out: object = value
            else:
                idx = g["idx"]
                raw_val = raw_values.iloc[idx]
                if hasattr(raw_val, "item"):
                    raw_val = raw_val.item()
                is_flag = pd.api.types.is_bool_dtype(dtypes.iloc[idx]) or source in FLAG_FEATURES
                display_value = _format_feature_value(source, raw_val, is_flag)
                value_out = bool(raw_val) if is_flag else (float(raw_val) if isinstance(raw_val, (int, float, np.number)) else str(raw_val))
            direction = "increased" if g["shap"] > 0 else "decreased"
            impact = _impact_label(abs(g["shap"]), max_abs)
            reasons.append({
                "rank": rank,
                "feature": source,
                "display_name": label,
                "value": value_out,
                "display_value": display_value,
                "shap_value": g["shap"],
                "direction": direction,
                "impact": impact,
                "sentence": f"{label}: {display_value} — {impact} {direction} the fraud risk score.",
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


# ---------------------------------------------------------------------------
# EX-01 helpers: map model columns back to the claim fields they came from,
# and give every field a label an investigator would actually use.
# ---------------------------------------------------------------------------
FLAG_FEATURES = {"is_highrisk_hobby", "is_exec_occupation", "is_new_customer", "is_major_damage"}

FEATURE_LABELS = {
    "is_major_damage": "Major damage incident",
    "incident_severity_ordinal": "Incident severity",
    "vehicle_claim_pct": "Vehicle share of total claim",
    "injury_claim_pct": "Injury share of total claim",
    "property_claim_pct": "Property share of total claim",
    "claim_to_premium_ratio": "Claim-to-premium ratio",
    "policy_age_at_incident_days": "Policy age at incident",
    "is_new_customer": "New customer (under 24 months)",
    "vehicle_age_at_incident": "Vehicle age at incident",
    "is_highrisk_hobby": "High-risk hobby (proxy feature)",
    "is_exec_occupation": "Executive occupation (proxy feature)",
    "months_as_customer": "Months as customer",
    "age": "Insured age",
    "policy_deductable": "Policy deductible",
    "policy_annual_premium": "Annual premium",
    "umbrella_limit": "Umbrella limit",
    "capital-gains": "Capital gains",
    "capital-loss": "Capital loss",
    "incident_hour_of_the_day": "Incident hour",
    "number_of_vehicles_involved": "Vehicles involved",
    "bodily_injuries": "Bodily injuries",
    "witnesses": "Number of witnesses",
    "total_claim_amount": "Total claim amount",
    "injury_claim": "Injury claim",
    "property_claim": "Property claim",
    "vehicle_claim": "Vehicle claim",
    "auto_year": "Vehicle model year",
    "policy_state": "Policy state",
    "policy_csl": "Policy CSL",
    "insured_sex": "Insured sex",
    "insured_education_level": "Education level",
    "insured_relationship": "Insured relationship",
    "incident_type": "Incident type",
    "collision_type": "Collision type",
    "authorities_contacted": "Authorities contacted",
    "incident_state": "Incident state",
    "property_damage": "Property damage",
    "police_report_available": "Police report available",
}

_MONEY = {"policy_annual_premium", "policy_deductable", "umbrella_limit", "capital-gains", "capital-loss",
          "total_claim_amount", "injury_claim", "property_claim", "vehicle_claim"}
_PCT = {"vehicle_claim_pct", "injury_claim_pct", "property_claim_pct"}
_SEVERITY_NAMES = {0: "Trivial Damage", 1: "Minor Damage", 2: "Major Damage", 3: "Total Loss"}


def build_source_map(feature_columns: list[str]) -> dict[str, tuple[str, str]]:
    """{model column -> (source claim field, category)} for every one-hot
    column. Longest prefix wins, so `incident_state_OH` maps to
    `incident_state`, never to some shorter `incident_` field."""
    from app.ml.feature_engineering import CATEGORICAL_COLUMNS
    prefixes = sorted(CATEGORICAL_COLUMNS, key=len, reverse=True)
    mapping = {}
    for col in feature_columns:
        for cat in prefixes:
            if col.startswith(cat + "_"):
                mapping[col] = (cat, col[len(cat) + 1:])
                break
    return mapping


def _format_feature_value(source: str, val, is_flag: bool) -> str:
    if is_flag:
        return "yes" if val else "no"
    try:
        fv = float(val)
    except (TypeError, ValueError):
        return str(val)
    if source == "incident_severity_ordinal":
        return _SEVERITY_NAMES.get(int(round(fv)), str(val))
    if source in _MONEY:
        return f"${fv:,.0f}"
    if source in _PCT:
        return f"{fv:.0%}"
    if source == "claim_to_premium_ratio":
        return f"{fv:.1f}x"
    if source == "policy_age_at_incident_days":
        return f"{fv:,.0f} days"
    if source == "vehicle_age_at_incident":
        return f"{fv:.0f} years"
    return _format_value(val)
