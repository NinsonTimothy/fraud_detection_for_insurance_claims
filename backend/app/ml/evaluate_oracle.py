"""
evaluate_oracle.py — scores the ALREADY-SHIPPED random_forest_final.pkl
against Oracle via oracle_adapter.py + the live feature pipeline (a genuine
external-validation test), AND separately trains fresh models directly on
Oracle's own real fields (a "stress test": is Oracle itself learnable
fraud data, or is the shipped model just bad at everything?). These are two
different questions — see docs/LIMITATIONS.md for why conflating them would
be misleading.

Run: python -m app.ml.evaluate_oracle   (from backend/)
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from app.ml.feature_engineering import align_to_training_columns, engineer_features
from app.ml.oracle_adapter import load_oracle_raw, map_oracle_to_raw_schema
from app.ml.psi import psi_report
from app.ml.uncertainty import bootstrap_metric_ci

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external" / "oracle"

# B8: ONE field-mapping table (raw field -> Oracle source, fidelity). Every
# count quoted anywhere is derived from this list, never typed. It mirrors
# oracle_adapter.map_oracle_to_raw_schema() exactly (asserted by a test).
FIELD_MAPPING = [
    ("age", "Age", "direct"),
    ("insured_sex", "Sex", "direct"),
    ("policy_deductable", "Deductible", "direct"),
    ("police_report_available", "PoliceReportFiled", "direct"),
    ("witnesses", "WitnessPresent (Yes/No -> 1/0; not a count)", "approximate"),
    ("number_of_vehicles_involved", "NumberOfCars (bucketed)", "approximate"),
    ("incident_date", "Year + MonthClaimed + WeekOfMonthClaimed", "approximate"),
    ("auto_year", "incident year - AgeOfVehicle (bucketed)", "approximate"),
    ("policy_bind_date", "incident_date - Days_Policy_Claim (bucketed)", "approximate"),
]
ORACLE_REAL_FIELDS = [f for f, _, _ in FIELD_MAPPING]


def field_mapping_table() -> pd.DataFrame:
    from app.ml.feature_engineering import RAW_FEATURE_COLUMNS
    mapped = {f: (src, fid) for f, src, fid in FIELD_MAPPING}
    return pd.DataFrame([{"raw_field": c, "oracle_source": mapped.get(c, ("—", "unmappable"))[0],
                          "fidelity": mapped.get(c, ("—", "unmappable"))[1]} for c in RAW_FEATURE_COLUMNS])


def univariate_auc_table(X_oracle: pd.DataFrame, y_oracle: pd.Series, variable_cols: list[str]) -> pd.DataFrame:
    """For every feature that still VARIES on Oracle: its univariate ROC-AUC
    on the development data and on Oracle. AUC near 0.5 on development data
    means the feature carried no signal to transfer in the first place;
    opposite sides of 0.5 mean the relationship reverses between datasets."""
    from app.ml.train import build_features, split_with_ids
    dev_df, test_df, y_dev, _ = split_with_ids()
    X_dev, _ = build_features(dev_df, test_df)
    rows = []
    for c in variable_cols:
        if c not in X_dev.columns or X_dev[c].nunique() < 2:
            continue
        a_dev = roc_auc_score(y_dev, X_dev[c].astype(float))
        a_or = roc_auc_score(y_oracle, X_oracle[c].astype(float))
        rows.append({"feature": c, "auc_development": a_dev, "auc_oracle": a_or,
                     "direction_reverses": bool((a_dev - 0.5) * (a_or - 0.5) < 0)})
    return pd.DataFrame(rows).sort_values("auc_development")


def evaluate_shipped_model_on_oracle():
    scaler = joblib.load(MODELS_DIR / "standard_scaler.pkl")
    # MS-01: evaluate the SHIPPED champion, whichever model won selection.
    champion_path = MODELS_DIR / "champion_model.pkl"
    rf_pipeline = joblib.load(champion_path if champion_path.exists() else MODELS_DIR / "random_forest_final.pkl")
    with open(MODELS_DIR / "feature_columns.json") as f:
        feature_columns = json.load(f)
    with open(MODELS_DIR / "metrics.json") as f:
        internal_metrics = json.load(f)
    operating_threshold = internal_metrics["operating_threshold"]

    oracle_raw = load_oracle_raw()
    mapped_df, y_oracle = map_oracle_to_raw_schema(oracle_raw)

    X_oracle = engineer_features(mapped_df)
    X_oracle = align_to_training_columns(X_oracle, feature_columns)
    X_oracle_scaled = pd.DataFrame(scaler.transform(X_oracle), columns=feature_columns)

    proba = rf_pipeline.predict_proba(X_oracle_scaled)[:, 1]
    pred = (proba >= operating_threshold).astype(int)

    roc = roc_auc_score(y_oracle, proba)
    pr = average_precision_score(y_oracle, proba)
    tp = int(((pred == 1) & (y_oracle == 1)).sum())
    fp = int(((pred == 1) & (y_oracle == 0)).sum())
    fn = int(((pred == 0) & (y_oracle == 1)).sum())
    tn = int(((pred == 0) & (y_oracle == 0)).sum())
    recall = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)

    # SH-04: bootstrap 95% CI for the Oracle numbers too, not just the
    # internal holdout — 15,420 rows is large, but the collapse itself
    # (recall/precision near 0) is exactly the kind of number worth
    # showing an interval around rather than a bare point estimate. See
    # uncertainty.py's module docstring.
    oracle_ci = bootstrap_metric_ci(y_oracle.values, proba, threshold=operating_threshold, n_boot=1000, random_state=42)

    # Which trained columns go constant on Oracle-mapped data?
    n_unique = X_oracle.nunique()
    constant_cols = set(n_unique[n_unique <= 1].index)
    shap_importance = pd.read_csv(PROCESSED_DIR / "shap_feature_importance.csv")
    shap_importance["constant_on_oracle"] = shap_importance["feature"].isin(constant_cols)
    constant_share = shap_importance.loc[shap_importance["constant_on_oracle"], "share_of_total"].sum()

    psi_cols = [c for c in ORACLE_REAL_FIELDS if c in X_oracle.columns] + \
        [c for c in X_oracle.columns if any(c.startswith(f) for f in ORACLE_REAL_FIELDS)]
    # PB-10 (fixed): this used to read risk_scores_test.csv, whose feature
    # columns are StandardScaler-SCALED (z-scores) — compared against
    # X_oracle's UNSCALED engineered features, that's a scale mismatch
    # that inflates PSI to meaningless values (reproduced: PSI("age")
    # came out ~6.9 comparing scaled-vs-unscaled, vs. ~0.05 — "no
    # significant shift" — comparing unscaled-vs-unscaled correctly).
    # psi_reference_features.csv holds the UNSCALED test features
    # specifically for this comparison, so both sides are on the same
    # scale (raw engineered feature values).
    internal_test = pd.read_csv(PROCESSED_DIR / "psi_reference_features.csv")
    psi_cols = [c for c in set(psi_cols) if c in internal_test.columns]
    psi_df = psi_report(internal_test, X_oracle, psi_cols) if psi_cols else pd.DataFrame()

    primary = internal_metrics.get("primary_model", "random_forest")
    primary_row = next(r for r in internal_metrics["model_comparison"] if r["model"] == primary)
    # OR-01: the one-word verdict is DERIVED from the CI, never typed. The
    # earlier docs called the result "random"; with a CI entirely below 0.5
    # that is wrong — the ranking is significantly inverted.
    roc_ci = oracle_ci.get("roc_auc", {})
    if roc_ci and roc_ci["ci_upper"] < 0.5:
        verdict = "significantly_inverted"
    elif roc_ci and roc_ci["ci_lower"] > 0.5:
        verdict = "better_than_random_but_degraded"
    else:
        verdict = "indistinguishable_from_random"
    variable_cols = sorted(set(X_oracle.columns) - constant_cols)
    univariate_auc_table(X_oracle, y_oracle, variable_cols).to_csv(EXTERNAL_DIR / "oracle_univariate_auc.csv", index=False)
    mapping = field_mapping_table()
    mapping.to_csv(EXTERNAL_DIR / "oracle_field_mapping.csv", index=False)
    report = {
        "model": primary,
        "field_mapping_counts": mapping["fidelity"].value_counts().to_dict(),
        "n_features_variable_on_oracle": len(variable_cols),
        "roc_auc_verdict": verdict,
        "internal_holdout_metrics": {
            "roc_auc": primary_row["roc_auc"],
            "pr_auc": primary_row["pr_auc"],
        },
        "oracle_metrics": {
            "roc_auc": float(roc), "pr_auc": float(pr), "recall": float(recall), "precision": float(precision),
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "operating_threshold": float(operating_threshold),
        },
        # SH-04: bootstrap 95% CI keyed the same as oracle_metrics above —
        # e.g. oracle_metrics_ci["roc_auc"] = {"ci_lower": ..., "ci_upper": ...}.
        "oracle_metrics_ci": oracle_ci,
        "oracle_n_rows": int(len(oracle_raw)),
        "oracle_fraud_rate": float(y_oracle.mean()),
        "n_features_total": len(feature_columns),
        "n_features_constant_on_oracle": int(len(constant_cols)),
        "share_of_shap_weight_constant_on_oracle": float(constant_share),
        "fields_mapped_for_real": ORACLE_REAL_FIELDS,
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
    }
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(EXTERNAL_DIR / "oracle_validation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    psi_df.to_csv(EXTERNAL_DIR / "oracle_psi_report.csv", index=False)
    shap_importance.to_csv(EXTERNAL_DIR / "oracle_constant_features.csv", index=False)

    sample = X_oracle_scaled.copy()
    sample["fraud_probability"] = proba
    sample["actual_fraud"] = y_oracle.values
    sample.sample(min(1000, len(sample)), random_state=42).to_csv(EXTERNAL_DIR / "oracle_scored_sample.csv", index=False)

    print(json.dumps(report, indent=2))
    return report


def train_fresh_oracle_models():
    """Stress test: train LR/RF/XGB fresh, directly on Oracle's own real
    fields — answers "is Oracle learnable at all", independent of whether
    THIS project's shipped model transfers to it."""
    oracle_raw = load_oracle_raw()
    y = (oracle_raw["FraudFound_P"] == 1).astype(int)

    categorical = ["Make", "AccidentArea", "Sex", "MaritalStatus", "Fault", "PolicyType",
                   "VehicleCategory", "VehiclePrice", "PoliceReportFiled", "WitnessPresent",
                   "AgentType", "AddressChange_Claim", "BasePolicy", "PastNumberOfClaims",
                   "AgeOfVehicle", "AgeOfPolicyHolder", "NumberOfSuppliments", "NumberOfCars"]
    numeric = ["WeekOfMonth", "Age", "RepNumber", "Deductible", "DriverRating", "Year"]

    X = pd.get_dummies(oracle_raw[categorical].astype(str), prefix=categorical)
    X[numeric] = oracle_raw[numeric]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = []
    fresh_models = {
        "oracle_lr": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "oracle_rf": RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced_subsample", random_state=42),
        "oracle_xgb": XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, eval_metric="logloss",
                                     scale_pos_weight=(y_train == 0).sum() / max(1, (y_train == 1).sum()), random_state=42),
    }
    for name, model in fresh_models.items():
        model.fit(X_train_s, y_train)
        proba = model.predict_proba(X_test_s)[:, 1]
        results.append({
            "model": name, "roc_auc": float(roc_auc_score(y_test, proba)),
            "pr_auc": float(average_precision_score(y_test, proba)),
        })
        joblib.dump(model, MODELS_DIR / f"{name}_final.pkl")

    df = pd.DataFrame(results)
    df.to_csv(PROCESSED_DIR / "oracle_model_comparison.csv", index=False)
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    print("=== Genuine external validation: shipped model vs. Oracle ===")
    evaluate_shipped_model_on_oracle()
    print("\n=== Stress test: fresh models trained directly on Oracle ===")
    train_fresh_oracle_models()
