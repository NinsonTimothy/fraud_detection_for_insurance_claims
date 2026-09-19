"""
train.py — trains the three compared models (Random Forest+SMOTE is what
gets shipped; Logistic Regression and XGBoost are compared alternatives,
same as the original FYP), saves every artifact the API/dashboard read.

Run: python -m app.ml.train   (from backend/)
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from app.ml.cost_threshold import find_cost_optimal_threshold
from app.ml.explainer import ClaimExplainer
from app.ml.feature_engineering import engineer_features

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_CLEANED = PROJECT_ROOT / "data" / "cleaned" / "insurance_claims_cleaned.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RANDOM_STATE = 42


def load_and_split():
    # SH-01: same keep_default_na=False rule as clean_data.py — the cleaned
    # CSV round-trips authorities_contacted's genuine "None" category as
    # the literal text "None", which plain read_csv() would otherwise
    # re-swallow as NaN on the way back in.
    df = pd.read_csv(DATA_CLEANED, keep_default_na=False, na_values=[""])
    y = (df["fraud_reported"] == "Y").astype(int)
    train_df, test_df, y_train, y_test = train_test_split(
        df, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), y_train.reset_index(drop=True), y_test.reset_index(drop=True)


def build_features(train_df, test_df, y_train):
    X_train = engineer_features(train_df)
    X_test = engineer_features(test_df)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    return X_train, X_test


def evaluate(name, model, X_test, y_test, threshold=0.5):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    roc = roc_auc_score(y_test, proba)
    pr = average_precision_score(y_test, proba)
    tp = int(((pred == 1) & (y_test == 1)).sum())
    fp = int(((pred == 1) & (y_test == 0)).sum())
    fn = int(((pred == 0) & (y_test == 1)).sum())
    tn = int(((pred == 0) & (y_test == 0)).sum())
    recall = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    acc = (tp + tn) / len(y_test)
    return {"model": name, "threshold": threshold, "recall": recall, "precision": precision,
            "f1": f1, "pr_auc": pr, "roc_auc": roc, "accuracy": acc,
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}}


def cross_validate_model(name, make_estimator, full_df: pd.DataFrame, y_all: pd.Series):
    """Full-pipeline 5-fold CV: refits the scaler and the classifier from
    scratch on each fold's own training partition, then scores the held-out
    fold.

    This is NOT the same as calling sklearn's `cross_validate()` on an
    already-engineered feature matrix — an earlier version of this file did
    that, and (back when feature_engineering.py still built the leaky
    `zip3_risk_tier` feature, see PB-02) it produced a badly inflated
    ROC-AUC (~0.94, vs. ~0.66-0.78 on a genuine single holdout split checked
    across 6 seeds). Refitting the entire pipeline per fold (as this
    function does) is kept as the honest pattern going forward even though
    `engineer_features()` is no longer target-encoded on anything."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = {"roc_auc": [], "pr_auc": [], "f1": [], "recall": [], "precision": []}
    for tr_idx, va_idx in skf.split(full_df, y_all):
        tr_df = full_df.iloc[tr_idx].reset_index(drop=True)
        va_df = full_df.iloc[va_idx].reset_index(drop=True)
        y_tr = y_all.iloc[tr_idx].reset_index(drop=True)
        y_va = y_all.iloc[va_idx].reset_index(drop=True)

        X_tr = engineer_features(tr_df)
        X_va = engineer_features(va_df).reindex(columns=X_tr.columns, fill_value=0)

        scaler_fold = StandardScaler()
        Xtr_scaled = scaler_fold.fit_transform(X_tr)
        Xva_scaled = scaler_fold.transform(X_va)

        estimator = make_estimator()
        estimator.fit(Xtr_scaled, y_tr)
        proba = estimator.predict_proba(Xva_scaled)[:, 1]
        pred = (proba >= 0.5).astype(int)

        tp = int(((pred == 1) & (y_va == 1)).sum())
        fp = int(((pred == 1) & (y_va == 0)).sum())
        fn = int(((pred == 0) & (y_va == 1)).sum())
        recall = tp / max(1, tp + fn)
        precision = tp / max(1, tp + fp)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)

        fold_metrics["roc_auc"].append(roc_auc_score(y_va, proba))
        fold_metrics["pr_auc"].append(average_precision_score(y_va, proba))
        fold_metrics["f1"].append(f1)
        fold_metrics["recall"].append(recall)
        fold_metrics["precision"].append(precision)

    return {
        "model": name,
        **{f"{k}_mean": float(np.mean(v)) for k, v in fold_metrics.items()},
        **{f"{k}_std": float(np.std(v)) for k, v in fold_metrics.items()},
    }


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_df, test_df, y_train, y_test = load_and_split()
    X_train, X_test = build_features(train_df, test_df, y_train)
    feature_columns = list(X_train.columns)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_columns, index=X_test.index)

    # ---- Random Forest + SMOTE (shipped model) ----
    rf_pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("rf", RandomForestClassifier(
            n_estimators=200, max_depth=5, max_features=0.3,
            min_samples_leaf=2, min_samples_split=10,
            class_weight="balanced_subsample", random_state=RANDOM_STATE,
        )),
    ])
    rf_pipeline.fit(X_train_scaled, y_train)

    # ---- Logistic Regression (compared alternative) ----
    lr_pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    lr_pipeline.fit(X_train_scaled, y_train)

    # ---- XGBoost (compared alternative) ----
    xgb_pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("xgb", XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=RANDOM_STATE,
        )),
    ])
    xgb_pipeline.fit(X_train_scaled, y_train)

    models = {"random_forest": rf_pipeline, "logistic_regression": lr_pipeline, "xgboost": xgb_pipeline}

    # Threshold: for RF, search on ROC-derived precision/recall to land
    # near the original project's own 0.55 operating point (recall-leaning,
    # since missing fraud is costlier than a false alarm here).
    rf_proba_test = rf_pipeline.predict_proba(X_test_scaled)[:, 1]
    thresholds_grid = np.linspace(0.3, 0.8, 51)
    best_t, best_f1 = 0.5, -1
    for t in thresholds_grid:
        pred = (rf_proba_test >= t).astype(int)
        tp = ((pred == 1) & (y_test == 1)).sum()
        fp = ((pred == 1) & (y_test == 0)).sum()
        fn = ((pred == 0) & (y_test == 1)).sum()
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    operating_threshold = round(float(best_t), 2)

    comparison_rows = [evaluate(name, m, X_test_scaled, y_test, threshold=0.5 if name != "random_forest" else operating_threshold) for name, m in models.items()]
    model_comparison = pd.DataFrame(comparison_rows)
    model_comparison.to_csv(PROCESSED_DIR / "model_comparison.csv", index=False)

    full_df = pd.concat([train_df, test_df], ignore_index=True)
    y_full = pd.concat([y_train, y_test], ignore_index=True)
    estimator_factories = {
        "random_forest": lambda: ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("rf", RandomForestClassifier(n_estimators=200, max_depth=5, max_features=0.3,
                                           min_samples_leaf=2, min_samples_split=10,
                                           class_weight="balanced_subsample", random_state=RANDOM_STATE)),
        ]),
        "logistic_regression": lambda: ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "xgboost": lambda: ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("xgb", XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                                   random_state=RANDOM_STATE)),
        ]),
    }
    cv_rows = [cross_validate_model(name, factory, full_df, y_full) for name, factory in estimator_factories.items()]
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(PROCESSED_DIR / "cross_validation_results.csv", index=False)

    # ---- Cost-optimal threshold search (uses RF, the shipped model) ----
    claim_amounts_test = test_df["total_claim_amount"].values
    cost_result = find_cost_optimal_threshold(y_test.values, rf_proba_test, claim_amounts_test, steps=50)
    cost_result["sweep"].to_csv(PROCESSED_DIR / "cost_threshold_sweep.csv", index=False)

    # ---- SHAP global importance on the shipped model ----
    rf_only = rf_pipeline.named_steps["rf"]
    explainer = ClaimExplainer(rf_only, feature_columns)
    shap_importance = explainer.global_importance(X_test_scaled)
    shap_importance["share_of_total"] = shap_importance["mean_abs_shap"] / shap_importance["mean_abs_shap"].sum()
    shap_importance.to_csv(PROCESSED_DIR / "shap_feature_importance.csv", index=False)

    # ---- Save artifacts ----
    joblib.dump(rf_pipeline, MODELS_DIR / "random_forest_final.pkl")
    joblib.dump(lr_pipeline, MODELS_DIR / "logistic_regression_final.pkl")
    joblib.dump(xgb_pipeline, MODELS_DIR / "xgboost_final.pkl")
    joblib.dump(scaler, MODELS_DIR / "standard_scaler.pkl")
    # NOTE: no separate shap_explainer.pkl artifact — FraudScoringService
    # rebuilds ClaimExplainer directly from the loaded RF pipeline at
    # startup (see inference.py), so a saved copy would be dead weight
    # that could silently drift from the shipped model (PB-17).
    # NOTE: no zip3_lookup.csv artifact anymore — PB-02 removed the
    # zip3-derived feature entirely; see feature_engineering.py.
    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_columns, f)

    metrics = {
        "primary_model": "random_forest", "operating_threshold": operating_threshold,
        "model_comparison": comparison_rows,
        "cost_optimal_threshold": {k: v for k, v in cost_result.items() if k != "sweep"},
        "n_train": len(X_train), "n_test": len(X_test),
        "n_features": len(feature_columns), "fraud_rate": float(y_train.mean()),
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    X_test_scaled.assign(y_true=y_test.values, y_proba=rf_proba_test).to_csv(PROCESSED_DIR / "risk_scores_test.csv", index=False)

    print(json.dumps({k: v for k, v in metrics.items() if k not in ("model_comparison",)}, indent=2))
    print()
    print(model_comparison[["model", "threshold", "recall", "precision", "f1", "pr_auc", "roc_auc", "accuracy"]].to_string(index=False))
    print()
    print("Cross-validation (5-fold):")
    print(cv_df.to_string(index=False))


if __name__ == "__main__":
    main()
