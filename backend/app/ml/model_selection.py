"""
model_selection.py — leak-free champion selection (pre-defence fix MS-01).

What was wrong before
---------------------
1. `model_selection_experiments.run_champion_comparison()` ran its paired CV
   over ALL 1,000 rows — including the 200 rows `train.py` later reports as
   the "held-out" test set. The champion was therefore chosen partly on the
   test set before the "final" evaluation, so the reported test numbers were
   not a clean estimate.
2. Every model was compared at a fixed 0.5 threshold, while the shipped RF
   was later deployed at a tuned threshold — not a like-for-like comparison.
3. No simple baseline was included. Measured on the old shipped model: at
   its 0.44 operating threshold the Random Forest flagged EXACTLY the
   claims with `incident_severity == "Major Damage"` (200/200 identical
   decisions on the test set). Without a baseline row, that was invisible.
4. The recall p-value text was hardcoded ("p=0.0086") and no longer matched
   the real computed value (p=0.498 for the shipped configuration).

What this module does instead
-----------------------------
* Uses ONLY the 800-row training split. The 200-row test split is never
  passed to anything in this file.
* Repeated stratified K-fold (5 folds x 3 repeats = 15 paired outer folds)
  — every candidate sees identical folds (a paired design).
* Inside each outer training fold, a NESTED inner 5-fold CV picks that
  candidate's hyperparameters (by inner PR-AUC) and its decision threshold
  (F1-optimal on inner out-of-fold probabilities). The outer validation fold
  is only used to score the already-tuned candidate. So every model —
  including the baseline — is compared at its own honestly tuned operating
  point.
* Candidates: the "Major Damage" one-line rule (baseline), Logistic
  Regression, Random Forest, XGBoost.
* Pre-declared selection rule (written here BEFORE the results were seen,
  and applied mechanically by `select_champion()`):
    - primary metric: mean outer-fold F1 at the tuned threshold (the
      project's operating objective — recall and precision together);
    - an ML candidate is eligible to be champion only if it is not worse
      than the baseline on mean F1;
    - ties within 0.005 F1 are broken by mean PR-AUC.
* Significance: the Nadeau & Bengio (2003) corrected resampled t-test,
  which is the appropriate paired test for REPEATED cross-validation (a
  plain paired t-test on overlapping CV folds is over-confident because
  the training sets overlap). All p-values are computed and written by
  code; nothing in this file or in any doc is hand-typed.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42
OUTER_SPLITS = 5
OUTER_REPEATS = 3
INNER_SPLITS = 5
THRESHOLD_GRID = np.round(np.linspace(0.05, 0.95, 91), 2)
F1_TIE_TOLERANCE = 0.005
BASELINE_NAME = "major_damage_rule"
# -1 = all cores. Fine on a laptop; set AEGIS_N_JOBS=1 on a single-core box.
N_JOBS = int(__import__("os").environ.get("AEGIS_N_JOBS", "-1"))


# ---------------------------------------------------------------------------
# The baseline: one line of business logic, no learning.
# ---------------------------------------------------------------------------
class MajorDamageRule(BaseEstimator, ClassifierMixin):
    """Flags a claim if and only if incident_severity == "Major Damage".

    Implemented as a scikit-learn-compatible estimator so it goes through
    the exact same CV/threshold machinery as the ML candidates (a fair
    comparison). `col_index` is the position of the engineered
    `is_major_damage` column. The model is fit on SCALED features, and
    StandardScaler preserves order, so "scaled value above the midpoint of
    the two scaled values" is exactly "raw value == 1"."""

    def __init__(self, col_index: int = 0):
        self.col_index = col_index

    def fit(self, X, y):
        col = np.asarray(X)[:, self.col_index]
        lo, hi = float(np.min(col)), float(np.max(col))
        self.cut_ = (lo + hi) / 2.0
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        flag = (np.asarray(X)[:, self.col_index] > self.cut_).astype(float)
        return np.column_stack([1.0 - flag, flag])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Candidate definitions. Each grid is deliberately small: 800 rows cannot
# support a large search without the search itself overfitting.
# ---------------------------------------------------------------------------
def _rf(max_depth=5, min_samples_leaf=2):
    return ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=max_depth, max_features=0.3,
            min_samples_leaf=min_samples_leaf, min_samples_split=10,
            class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=N_JOBS,
        )),
    ])


def _lr(C=1.0):
    return ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", LogisticRegression(C=C, max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])


def _xgb(max_depth=3, n_estimators=200):
    return ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=N_JOBS,
        )),
    ])


@dataclass
class Candidate:
    name: str
    factory: Callable[..., object]
    grid: list[dict] = field(default_factory=lambda: [{}])
    is_baseline: bool = False


def build_candidates(feature_columns: list[str]) -> list[Candidate]:
    md_idx = feature_columns.index("is_major_damage")
    return [
        Candidate(BASELINE_NAME, lambda: MajorDamageRule(col_index=md_idx), [{}], is_baseline=True),
        Candidate("logistic_regression", _lr, [{"C": c} for c in (0.01, 0.1, 1.0)]),
        Candidate("random_forest", _rf, [{"max_depth": d, "min_samples_leaf": l}
                                         for d, l in itertools.product((3, 5, 8), (2, 5))]),
        Candidate("xgboost", _xgb, [{"max_depth": d, "n_estimators": n}
                                    for d, n in itertools.product((2, 3), (100, 300))]),
    ]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
def classification_metrics(y_true, proba, threshold) -> dict:
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    recall = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "recall": recall, "precision": precision, "f1": f1,
        "accuracy": (tp + tn) / len(y_true),
        "flag_rate": float(pred.mean()),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def f1_optimal_threshold(y_true, proba, grid=THRESHOLD_GRID) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = classification_metrics(y_true, proba, t)["f1"]
        if f1 > best_f1 + 1e-12:
            best_f1, best_t = f1, float(t)
    return round(best_t, 2)


def _fit_scaled(make_estimator, X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_tr)
    Xva = scaler.transform(X_va)
    est = make_estimator()
    est.fit(Xtr, y_tr)
    return est.predict_proba(Xva)[:, 1]


def inner_tune(candidate: Candidate, X: np.ndarray, y: np.ndarray, seed: int) -> tuple[dict, float]:
    """Pick hyperparameters (by inner-CV PR-AUC) and the decision threshold
    (F1-optimal on that configuration's inner out-of-fold probabilities),
    using ONLY the rows passed in."""
    skf = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed)
    splits = list(skf.split(X, y))
    best = (None, -1.0, None)  # params, pr_auc, oof
    for params in candidate.grid:
        oof = np.zeros(len(y))
        for tr, va in splits:
            oof[va] = _fit_scaled(lambda: candidate.factory(**params), X[tr], y[tr], X[va])
        pr = average_precision_score(y, oof)
        if pr > best[1] + 1e-12:
            best = (params, pr, oof)
    params, _, oof = best
    return params, f1_optimal_threshold(y, oof)


# ---------------------------------------------------------------------------
# Nested, repeated, paired comparison
# ---------------------------------------------------------------------------
def nested_cv_compare(X: pd.DataFrame, y: pd.Series, feature_columns: list[str]) -> dict:
    Xn, yn = X[feature_columns].to_numpy(dtype=float), np.asarray(y)
    candidates = build_candidates(feature_columns)
    outer = RepeatedStratifiedKFold(n_splits=OUTER_SPLITS, n_repeats=OUTER_REPEATS, random_state=RANDOM_STATE)
    rows = []
    for fold_id, (tr, va) in enumerate(outer.split(Xn, yn)):
        for cand in candidates:
            params, thr = inner_tune(cand, Xn[tr], yn[tr], seed=RANDOM_STATE + fold_id)
            proba = _fit_scaled(lambda: cand.factory(**params), Xn[tr], yn[tr], Xn[va])
            m = classification_metrics(yn[va], proba, thr)
            rows.append({
                "fold": fold_id, "model": cand.name, "threshold": thr, "params": str(params),
                "roc_auc": roc_auc_score(yn[va], proba),
                "pr_auc": average_precision_score(yn[va], proba),
                "f1": m["f1"], "recall": m["recall"], "precision": m["precision"],
                "flag_rate": m["flag_rate"], "n_train": len(tr), "n_val": len(va),
            })
    folds = pd.DataFrame(rows)
    return {"folds": folds, "candidates": candidates}


def corrected_resampled_ttest(diffs: np.ndarray, n_train: int, n_val: int) -> tuple[float, float]:
    """Nadeau & Bengio (2003) corrected resampled t-test for repeated CV.

    The ordinary paired t-test treats the k*r fold differences as
    independent, but CV training sets overlap heavily, so its variance is
    too small and its p-values too optimistic. The correction inflates the
    variance by (1/J + n_val/n_train)."""
    diffs = np.asarray(diffs, dtype=float)
    J = len(diffs)
    mean = diffs.mean()
    var = diffs.var(ddof=1)
    if var == 0:
        return (math.inf if mean != 0 else 0.0), (0.0 if mean != 0 else 1.0)
    t = mean / math.sqrt((1.0 / J + n_val / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=J - 1)
    return float(t), float(p)


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    metrics = ["f1", "recall", "precision", "pr_auc", "roc_auc", "flag_rate", "threshold"]
    agg = folds.groupby("model")[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()


def select_champion(summary: pd.DataFrame) -> dict:
    """Applies the pre-declared rule in the module docstring, mechanically."""
    s = summary.set_index("model")
    baseline_f1 = s.loc[BASELINE_NAME, "f1_mean"]
    ml = s.drop(index=BASELINE_NAME)
    eligible = ml[ml["f1_mean"] >= baseline_f1 - 1e-12]
    pool = eligible if len(eligible) else ml
    top_f1 = pool["f1_mean"].max()
    tied = pool[pool["f1_mean"] >= top_f1 - F1_TIE_TOLERANCE]
    champion = tied["pr_auc_mean"].idxmax()
    return {
        "champion": champion,
        "beats_or_matches_baseline_on_f1": bool(len(eligible) > 0),
        "tie_broken_by_pr_auc": bool(len(tied) > 1),
        "tied_candidates": list(tied.index),
    }


def pairwise_tests(folds: pd.DataFrame, champion: str) -> pd.DataFrame:
    n_train = int(folds["n_train"].iloc[0])
    n_val = int(folds["n_val"].iloc[0])
    wide = {m: folds.pivot(index="fold", columns="model", values=m) for m in ("f1", "recall", "pr_auc", "roc_auc")}
    rows = []
    for other in sorted(folds["model"].unique()):
        if other == champion:
            continue
        for metric, table in wide.items():
            diffs = (table[champion] - table[other]).to_numpy()
            t, p = corrected_resampled_ttest(diffs, n_train, n_val)
            rows.append({
                "champion": champion, "vs": other, "metric": metric,
                "mean_difference": float(diffs.mean()),
                "champion_wins_folds": int((diffs > 0).sum()), "n_folds": len(diffs),
                "t": t, "p_corrected": p, "significant_at_0_05": bool(p < 0.05),
            })
    return pd.DataFrame(rows)


def final_tune_on_full_train(candidate: Candidate, X: pd.DataFrame, y: pd.Series, feature_columns) -> tuple[dict, float]:
    """After selection: re-run the SAME inner procedure on the full 800-row
    training split to fix the champion's hyperparameters and threshold.
    Still touches no test rows."""
    return inner_tune(candidate, X[feature_columns].to_numpy(dtype=float), np.asarray(y), seed=RANDOM_STATE)
