"""
model_selection_experiments.py — SH-03 evidence: does this project need
BOTH SMOTE oversampling AND class_weight="balanced"/"balanced_subsample"
(what train.py currently does for every model), or is one of the two
enough? Also runs PB-14's champion-selection evidence: a paired (same
5 folds across every candidate), refit-per-fold CV comparison of Random
Forest / Logistic Regression / XGBoost with a paired significance test on
the project's own top-priority metrics (Recall, then F1 — see README.md's
"Success metrics, in priority order"), used to check D4's default
champion (regularised Logistic Regression) against measured evidence.

This is intentionally a separate module from train.py: it runs several
EXTRA CV passes purely to justify modeling choices and is not needed to
ship the model artifacts train.py produces, so keeping it out of the main
training path keeps `python -m app.ml.train` fast and focused.

Run: python -m app.ml.model_selection_experiments   (from backend/)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from app.ml.feature_engineering import engineer_features

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_CLEANED = PROJECT_ROOT / "data" / "cleaned" / "insurance_claims_cleaned.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RANDOM_STATE = 42
N_SPLITS = 5


def _paired_cv(name: str, make_estimator, X: pd.DataFrame, y: pd.Series) -> dict:
    """Same refit-per-fold pattern as train.py's cross_validate_model(),
    reused here so every variant in the comparison is scored on the exact
    same folds (paired comparison, not just similarly-seeded ones)."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = {"roc_auc": [], "pr_auc": [], "f1": [], "recall": [], "precision": []}
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        scaler = StandardScaler()
        Xtr_scaled = scaler.fit_transform(X_tr)
        Xva_scaled = scaler.transform(X_va)

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
        "variant": name,
        **{f"{k}_mean": float(np.mean(v)) for k, v in fold_metrics.items()},
        **{f"{k}_std": float(np.std(v)) for k, v in fold_metrics.items()},
        "_fold_recall": fold_metrics["recall"], "_fold_f1": fold_metrics["f1"],
    }


def run_smote_vs_classweight() -> pd.DataFrame:
    # MS-01: TRAIN split only (same split as train.py) — this experiment
    # informs a modelling choice, so it must not see the test rows either.
    from app.ml.train import load_and_split
    train_df, _test_df, y, _y_test = load_and_split()
    X = engineer_features(train_df)

    variants = {
        "rf_smote_and_classweight (previous build)": lambda: ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("rf", RandomForestClassifier(n_estimators=200, max_depth=5, max_features=0.3,
                                           min_samples_leaf=2, min_samples_split=10,
                                           class_weight="balanced_subsample", random_state=RANDOM_STATE)),
        ]),
        "rf_smote_only": lambda: ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("rf", RandomForestClassifier(n_estimators=200, max_depth=5, max_features=0.3,
                                           min_samples_leaf=2, min_samples_split=10,
                                           class_weight=None, random_state=RANDOM_STATE)),
        ]),
        "rf_classweight_only (shipped approach)": lambda: SkPipeline([
            ("rf", RandomForestClassifier(n_estimators=200, max_depth=5, max_features=0.3,
                                           min_samples_leaf=2, min_samples_split=10,
                                           class_weight="balanced_subsample", random_state=RANDOM_STATE)),
        ]),
        "lr_smote_and_classweight (previous build)": lambda: ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "lr_smote_only": lambda: ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("lr", LogisticRegression(max_iter=2000, class_weight=None, random_state=RANDOM_STATE)),
        ]),
        "lr_classweight_only (shipped approach)": lambda: SkPipeline([
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
    }

    rows = [_paired_cv(name, factory, X, y) for name, factory in variants.items()]
    result_df = pd.DataFrame(rows).drop(columns=["_fold_recall", "_fold_f1"])
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(PROCESSED_DIR / "smote_vs_classweight_comparison.csv", index=False)
    print(result_df.to_string(index=False))
    return result_df


# MS-01 (pre-defence fix): `run_champion_comparison()` was REMOVED from this
# file. It (a) cross-validated on all 1,000 rows, so the 200 test rows were
# used to pick the champion, (b) compared models at a fixed 0.5 threshold,
# (c) had no simple baseline, and (d) wrote a hardcoded note claiming
# a recall p-value from an older configuration instead of the value it
# actually computed for the shipped configuration (not significant). Champion selection now lives
# in `app/ml/model_selection.py` and runs inside `python -m app.ml.train`;
# every p-value is computed and written by code (champion_pairwise_tests.csv,
# champion_decision.json) and never typed into prose.


if __name__ == "__main__":
    print("=== SH-03: SMOTE vs. class_weight vs. both, paired 5-fold CV ===")
    run_smote_vs_classweight()
    print()
    print("Champion selection: run `python -m app.ml.train` (see app/ml/model_selection.py).")
