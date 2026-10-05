"""
model_selection.py — leak-free, pre-declared champion selection (B1/B2/B3).

DATA: only the 800 development rows. The 200-row test split is never passed
to anything in this module; `nested_cv_compare()` records the row ids it saw
so a test can assert that (D3).

IMBALANCE (B2): class weighting only. SMOTE was removed: the development-only
comparison in model_selection_experiments.run_smote_vs_classweight() shows it
adds nothing beyond class weighting (artifact: smote_vs_classweight_comparison.csv).

CANDIDATES (fixed hyperparameters, declared here, not tuned on results):
  major_damage_rule   — baseline: flag iff incident_severity == "Major Damage"
  logistic_regression — L2, C=1.0, class_weight="balanced"            complexity 1
  random_forest       — 200 trees, max_depth 5, class_weight balanced  complexity 2
  xgboost             — 200 trees, depth 3, lr 0.05, scale_pos_weight  complexity 3

PROTOCOL: repeated stratified K-fold (CV_SPLITS x CV_REPEATS, default 5 x 10 =
50 paired folds). Each candidate gets its decision threshold by the IDENTICAL
procedure: F1-optimal threshold on inner 5-fold out-of-fold predictions of the
outer training part, then applied to the outer validation part.

SELECTION RULE (written before any result was seen; applied mechanically by
`select_champion()`):
  1. Among the ML candidates, `best` = highest mean PR-AUC.
  2. Walk the ML candidates from simplest to most complex (LR, RF, XGB). The
     champion is the FIRST candidate that `best` does NOT beat significantly
     on PR-AUC (Nadeau-Bengio corrected resampled t-test, two-sided,
     alpha = SIGNIFICANCE_ALPHA). `best` itself always qualifies, so a
     champion always exists.
  3. The rule baseline is not eligible to be champion (it cannot rank claims
     within a severity level or be explained per claim), but the champion is
     tested against it on PR-AUC, recall and F1 and the result is reported
     plainly — including when no ML model beats it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from app.core.config import CV_REPEATS, CV_SPLITS, SIGNIFICANCE_ALPHA

RANDOM_STATE = 42
INNER_SPLITS = 5
THRESHOLD_GRID = np.round(np.linspace(0.02, 0.98, 97), 2)
BASELINE_NAME = "major_damage_rule"
N_JOBS = int(__import__("os").environ.get("AEGIS_N_JOBS", "-1"))


class MajorDamageRule(BaseEstimator, ClassifierMixin):
    """Score 1 if is_major_damage else 0. Works on scaled features because
    StandardScaler preserves order (scaled > midpoint <=> raw == 1)."""

    def __init__(self, col_index: int = 0):
        self.col_index = col_index

    def fit(self, X, y):
        col = np.asarray(X, dtype=float)[:, self.col_index]
        self.cut_ = (float(col.min()) + float(col.max())) / 2.0
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        flag = (np.asarray(X, dtype=float)[:, self.col_index] > self.cut_).astype(float)
        return np.column_stack([1.0 - flag, flag])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class BalancedXGB(XGBClassifier):
    """XGBoost with scale_pos_weight = negatives/positives of the data it is
    fit on — the gradient-boosting equivalent of class_weight="balanced"."""

    def fit(self, X, y, **kw):
        y = np.asarray(y)
        self.set_params(scale_pos_weight=float((y == 0).sum()) / max(1.0, float((y == 1).sum())))
        return super().fit(X, y, **kw)


def make_lr():
    return LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE)


def make_rf():
    return RandomForestClassifier(n_estimators=200, max_depth=5, max_features=0.3, min_samples_leaf=2,
                                  min_samples_split=10, class_weight="balanced_subsample",
                                  random_state=RANDOM_STATE, n_jobs=N_JOBS)


def make_xgb():
    return BalancedXGB(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                       eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=N_JOBS)


@dataclass
class Candidate:
    name: str
    factory: Callable[[], object]
    complexity: int
    is_baseline: bool = False


def build_candidates(feature_columns: list[str]) -> list[Candidate]:
    md = feature_columns.index("is_major_damage")
    return [
        Candidate(BASELINE_NAME, lambda: MajorDamageRule(col_index=md), 0, is_baseline=True),
        Candidate("logistic_regression", make_lr, 1),
        Candidate("random_forest", make_rf, 2),
        Candidate("xgboost", make_xgb, 3),
    ]


# --------------------------------------------------------------- helpers ---
def classification_metrics(y_true, proba, threshold) -> dict:
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum()); fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum()); tn = int(((pred == 0) & (y_true == 0)).sum())
    recall = tp / max(1, tp + fn); precision = tp / max(1, tp + fp)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {"recall": recall, "precision": precision, "f1": f1, "accuracy": (tp + tn) / len(y_true),
            "flag_rate": float(pred.mean()), "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}}


def f1_optimal_threshold(y_true, proba, grid=THRESHOLD_GRID) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = classification_metrics(y_true, proba, t)["f1"]
        if f1 > best_f1 + 1e-12:
            best_f1, best_t = f1, float(t)
    return round(best_t, 2)


def fit_predict(make_estimator, X_tr, y_tr, X_va):
    """Scaler + estimator refit on the training part only."""
    scaler = StandardScaler().fit(X_tr)
    est = make_estimator()
    est.fit(scaler.transform(X_tr), y_tr)
    return est.predict_proba(scaler.transform(X_va))[:, 1]


def oof_proba(make_estimator, X: np.ndarray, y: np.ndarray, n_splits: int = INNER_SPLITS, seed: int = RANDOM_STATE) -> np.ndarray:
    oof = np.zeros(len(y))
    for tr, va in StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y):
        oof[va] = fit_predict(make_estimator, X[tr], y[tr], X[va])
    return oof


# ------------------------------------------------------------ comparison ---
def nested_cv_compare(X_dev: pd.DataFrame, y_dev: pd.Series, feature_columns: list[str]) -> dict:
    Xn = X_dev[feature_columns].to_numpy(dtype=float)
    yn = np.asarray(y_dev)
    candidates = build_candidates(feature_columns)
    outer = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE)
    rows = []
    for fold, (tr, va) in enumerate(outer.split(Xn, yn)):
        for c in candidates:
            if c.is_baseline:
                thr = 0.5  # the rule outputs only 0/1; every threshold in (0,1) is identical
            else:
                thr = f1_optimal_threshold(yn[tr], oof_proba(c.factory, Xn[tr], yn[tr], seed=RANDOM_STATE + fold))
            proba = fit_predict(c.factory, Xn[tr], yn[tr], Xn[va])
            m = classification_metrics(yn[va], proba, thr)
            rows.append({"fold": fold, "model": c.name, "threshold": thr,
                         "roc_auc": roc_auc_score(yn[va], proba), "pr_auc": average_precision_score(yn[va], proba),
                         "f1": m["f1"], "recall": m["recall"], "precision": m["precision"],
                         "flag_rate": m["flag_rate"], "n_train": len(tr), "n_val": len(va)})
    return {"folds": pd.DataFrame(rows), "candidates": candidates,
            "row_ids_seen": sorted(int(i) for i in X_dev.index)}


def corrected_resampled_ttest(diffs, n_train: int, n_val: int) -> tuple[float, float]:
    """Nadeau & Bengio (2003): variance inflated by (1/J + n_val/n_train)
    because CV training sets overlap."""
    d = np.asarray(diffs, dtype=float)
    J, mean, var = len(d), float(d.mean()), float(d.var(ddof=1))
    if var == 0:
        return (math.inf if mean else 0.0), (0.0 if mean else 1.0)
    t = mean / math.sqrt((1.0 / J + n_val / n_train) * var)
    return float(t), float(2 * stats.t.sf(abs(t), df=J - 1))


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    agg = folds.groupby("model")[["pr_auc", "roc_auc", "f1", "recall", "precision", "flag_rate", "threshold"]].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()


def pairwise_test(folds: pd.DataFrame, a: str, b: str, metric: str) -> dict:
    wide = folds.pivot(index="fold", columns="model", values=metric)
    diffs = (wide[a] - wide[b]).to_numpy()
    t, p = corrected_resampled_ttest(diffs, int(folds["n_train"].iloc[0]), int(folds["n_val"].iloc[0]))
    return {"a": a, "b": b, "metric": metric, "mean_difference": float(diffs.mean()),
            "a_wins_folds": int((diffs > 0).sum()), "n_folds": int(len(diffs)), "t": t, "p_corrected": p,
            "significant": bool(p < SIGNIFICANCE_ALPHA)}


def all_pairwise_tests(folds: pd.DataFrame) -> pd.DataFrame:
    models = sorted(folds["model"].unique())
    rows = [pairwise_test(folds, a, b, m) for i, a in enumerate(models) for b in models[i + 1:]
            for m in ("pr_auc", "recall", "f1")]
    return pd.DataFrame(rows)


def select_champion(folds: pd.DataFrame, candidates: list[Candidate]) -> dict:
    """The pre-declared rule in the module docstring — no other logic."""
    ml = sorted([c for c in candidates if not c.is_baseline], key=lambda c: c.complexity)
    means = folds.groupby("model")["pr_auc"].mean()
    best = max(ml, key=lambda c: means[c.name]).name
    trail = []
    champion = best
    for c in ml:
        if c.name == best:
            trail.append({"candidate": c.name, "decision": "best PR-AUC — qualifies"})
            champion = best
            break
        test = pairwise_test(folds, best, c.name, "pr_auc")
        trail.append({"candidate": c.name, "vs_best_p": test["p_corrected"],
                      "decision": "best is significantly better — skip" if test["significant"]
                      else "best is NOT significantly better — choose this simpler model"})
        if not test["significant"]:
            champion = c.name
            break
    vs_rule = {m: pairwise_test(folds, champion, BASELINE_NAME, m) for m in ("pr_auc", "recall", "f1")}
    return {
        "best_pr_auc_model": best,
        "champion": champion,
        "decision_trail": trail,
        "champion_vs_rule": vs_rule,
        "champion_significantly_beats_rule_on": [m for m, r in vs_rule.items() if r["significant"] and r["mean_difference"] > 0],
        "rule_significantly_beats_champion_on": [m for m, r in vs_rule.items() if r["significant"] and r["mean_difference"] < 0],
    }
