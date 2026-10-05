"""
train.py — trains the three compared models (Random Forest+SMOTE is what
gets shipped; Logistic Regression and XGBoost are compared alternatives,
same as the original FYP), saves every artifact the API/dashboard read.

PB-03 (fixed): the operating threshold, the cost-optimal threshold, and
the global SHAP importance used to all be computed directly from
`X_test_scaled`/`y_test` — the same rows `evaluate()` then reports final
performance numbers against. That's leakage: picking the threshold that
maximizes F1 ON the test set and then reporting that threshold's F1 ON
the same test set is optimistic by construction, not an honest estimate
of how the model would perform at a threshold chosen without having seen
those rows. Fixed by choosing every threshold (and computing SHAP) from
honest out-of-fold predictions on the TRAINING set only (see
`_out_of_fold_proba()` below) — the true test set (`X_test_scaled`/
`y_test`) is now touched exactly once, in `evaluate()`, purely to REPORT
performance at a threshold that was chosen without it.

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

from app.core.config import INCLUDE_PROXY_FEATURES
from app.ml.cost_threshold import find_cost_optimal_threshold
from app.ml import model_selection as ms
from app.ml.explainer import ClaimExplainer, build_source_map
from app.ml.feature_engineering import engineer_features
from app.ml.uncertainty import bootstrap_metric_ci

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


def build_features(train_df, test_df, y_train, include_proxy_features: bool | None = None):
    X_train = engineer_features(train_df, include_proxy_features=include_proxy_features)
    X_test = engineer_features(test_df, include_proxy_features=include_proxy_features)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    return X_train, X_test


def _out_of_fold_proba(make_estimator, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> np.ndarray:
    """PB-03: honest out-of-fold predicted probabilities for every row in
    `X`/`y`, refitting a FRESH scaler + estimator on each fold's own
    training partition and predicting only on that fold's held-out rows —
    the true test set is never involved. Used to choose the operating
    threshold, the cost-optimal threshold, and (via the RF factory) to
    compute SHAP importance, all without letting the test set influence
    any of those choices. Same refit-per-fold pattern as
    `cross_validate_model()`, just returning row-level probabilities
    instead of aggregated fold metrics."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(X))
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr = y.iloc[tr_idx]
        scaler_fold = StandardScaler()
        Xtr_scaled = scaler_fold.fit_transform(X_tr)
        Xva_scaled = scaler_fold.transform(X_va)
        estimator = make_estimator()
        estimator.fit(Xtr_scaled, y_tr)
        oof[va_idx] = estimator.predict_proba(Xva_scaled)[:, 1]
    return oof


def _f1_optimal_threshold(y_true: np.ndarray, proba: np.ndarray, grid: np.ndarray) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        pred = (proba >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return round(float(best_t), 2)


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


def cross_validate_model(name, make_estimator, full_df: pd.DataFrame, y_all: pd.Series, include_proxy_features: bool | None = None):
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

        X_tr = engineer_features(tr_df, include_proxy_features=include_proxy_features)
        X_va = engineer_features(va_df, include_proxy_features=include_proxy_features).reindex(columns=X_tr.columns, fill_value=0)

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


def run_proxy_feature_ablation(train_df, test_df, y_train, y_test, full_df, y_full) -> pd.DataFrame:
    """SH-02 / D3: report BOTH proxy-feature variants (measured, not
    asserted) — `is_highrisk_hobby`/`is_exec_occupation` on vs. off — on
    the champion architecture (RF+SMOTE, same hyperparameters either way).
    `INCLUDE_PROXY_FEATURES` (app.core.config, default False) decides
    which variant actually ships as random_forest_final.pkl; this function
    only quantifies what the excluded variant would have changed, for the
    record (data/processed/proxy_feature_ablation.csv,
    docs/REBUILD_NOTES.md). Both a single holdout comparison (threshold
    0.5 for both, so the comparison isn't confounded by two different
    threshold choices) and a full 5-fold refit-per-fold CV are reported,
    since a single 200-row holdout split is a noisy way to compare a
    2-feature difference on its own."""
    def make_rf():
        return ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("rf", RandomForestClassifier(
                n_estimators=200, max_depth=5, max_features=0.3,
                min_samples_leaf=2, min_samples_split=10,
                class_weight="balanced_subsample", random_state=RANDOM_STATE,
            )),
        ])

    rows = []
    for variant, include in (("proxy_features_off", False), ("proxy_features_on", True)):
        X_tr, X_te = build_features(train_df, test_df, y_train, include_proxy_features=include)
        scaler_v = StandardScaler()
        X_tr_scaled = pd.DataFrame(scaler_v.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
        X_te_scaled = pd.DataFrame(scaler_v.transform(X_te), columns=X_tr.columns, index=X_te.index)
        rf = make_rf()
        rf.fit(X_tr_scaled, y_train)
        holdout = evaluate(variant, rf, X_te_scaled, y_test, threshold=0.5)
        cv = cross_validate_model(variant, make_rf, full_df, y_full, include_proxy_features=include)
        rows.append({
            "variant": variant, "include_proxy_features": include, "n_features": len(X_tr.columns),
            "holdout_recall": holdout["recall"], "holdout_precision": holdout["precision"],
            "holdout_f1": holdout["f1"], "holdout_pr_auc": holdout["pr_auc"], "holdout_roc_auc": holdout["roc_auc"],
            "cv_recall_mean": cv["recall_mean"], "cv_recall_std": cv["recall_std"],
            "cv_f1_mean": cv["f1_mean"], "cv_f1_std": cv["f1_std"],
            "cv_pr_auc_mean": cv["pr_auc_mean"], "cv_pr_auc_std": cv["pr_auc_std"],
            "cv_roc_auc_mean": cv["roc_auc_mean"], "cv_roc_auc_std": cv["roc_auc_std"],
        })
    return pd.DataFrame(rows)


def _final_estimator(pipeline):
    """The bare classifier at the end of a pipeline (what SHAP explains)."""
    return pipeline.steps[-1][1] if hasattr(pipeline, "steps") else pipeline


def main():
    """MS-01 (pre-defence fix): champion selection now happens ENTIRELY on
    the 800-row training split (nested, repeated, paired CV in
    model_selection.py), with a "Major Damage" one-line rule included as a
    baseline. The 200-row test split is used exactly once, at the end, to
    report every candidate at the threshold it chose without seeing it."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_df, test_df, y_train, y_test = load_and_split()
    X_train, X_test = build_features(train_df, test_df, y_train)
    feature_columns = list(X_train.columns)

    # ---- 1. Model selection on TRAIN ONLY ----
    print("Running nested, repeated CV on the training split (this is the slow step)...")
    selection = ms.nested_cv_compare(X_train, y_train, feature_columns)
    folds = selection["folds"]
    candidates = {c.name: c for c in selection["candidates"]}
    folds.to_csv(PROCESSED_DIR / "model_selection_folds.csv", index=False)
    summary = ms.summarize(folds)
    summary.to_csv(PROCESSED_DIR / "model_selection_summary.csv", index=False)
    # cross_validation_results.csv keeps its old column layout for the
    # dashboard, but is now the TRAIN-ONLY nested-CV summary (it used to be
    # a CV over all 1,000 rows, test set included).
    cv_df = summary[["model"] + [c for c in summary.columns if c.endswith(("_mean", "_std"))]]
    cv_df.to_csv(PROCESSED_DIR / "cross_validation_results.csv", index=False)

    decision = ms.select_champion(summary)
    champion_name = decision["champion"]
    tests_df = ms.pairwise_tests(folds, champion_name)
    tests_df.to_csv(PROCESSED_DIR / "champion_pairwise_tests.csv", index=False)

    # ---- 2. Fix every candidate's hyperparameters + threshold on the full
    # training split (same inner procedure, still no test rows) ----
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_columns, index=X_test.index)

    fitted, chosen = {}, {}
    for name, cand in candidates.items():
        params, thr = ms.final_tune_on_full_train(cand, X_train, y_train, feature_columns)
        model = cand.factory(**params)
        model.fit(X_train_scaled, y_train)
        fitted[name], chosen[name] = model, {"params": params, "threshold": thr}

    # ---- 3. The ONE look at the test set: report, never choose ----
    comparison_rows = []
    test_proba = {}
    for name, model in fitted.items():
        proba = model.predict_proba(X_test_scaled)[:, 1]
        test_proba[name] = proba
        row = evaluate(name, model, X_test_scaled, y_test, threshold=chosen[name]["threshold"])
        row["hyperparameters"] = chosen[name]["params"]
        row["is_baseline"] = candidates[name].is_baseline
        row["flag_rate"] = float((proba >= chosen[name]["threshold"]).mean())
        comparison_rows.append(row)
    # champion first, then the rest, baseline last — readable tables
    order = [champion_name] + [n for n in fitted if n not in (champion_name, ms.BASELINE_NAME)] + [ms.BASELINE_NAME]
    comparison_rows = sorted(comparison_rows, key=lambda r: order.index(r["model"]))
    pd.DataFrame(comparison_rows).drop(columns=["confusion_matrix", "hyperparameters"]).to_csv(
        PROCESSED_DIR / "model_comparison.csv", index=False)

    bootstrap_rows = []
    for row in comparison_rows:
        name = row["model"]
        ci = bootstrap_metric_ci(y_test.values, test_proba[name], threshold=chosen[name]["threshold"],
                                 n_boot=1000, random_state=RANDOM_STATE)
        for metric, interval in ci.items():
            bootstrap_rows.append({
                "model": name, "metric": metric, "point_estimate": row[metric],
                "ci_lower": interval["ci_lower"], "ci_upper": interval["ci_upper"],
                "n_boot_effective": interval["n_boot_effective"],
            })
    pd.DataFrame(bootstrap_rows).to_csv(PROCESSED_DIR / "holdout_bootstrap_ci.csv", index=False)

    # ---- 4. How different is the champion from the one-line rule? ----
    champ_flag = test_proba[champion_name] >= chosen[champion_name]["threshold"]
    rule_flag = test_proba[ms.BASELINE_NAME] >= chosen[ms.BASELINE_NAME]["threshold"]
    baseline_agreement = {
        "test_decision_agreement": float((champ_flag == rule_flag).mean()),
        "flagged_by_champion_not_rule": int((champ_flag & ~rule_flag).sum()),
        "flagged_by_rule_not_champion": int((~champ_flag & rule_flag).sum()),
        "n_test": int(len(y_test)),
        "note": ("Share of test claims on which the champion's flag/no-flag decision is identical "
                 "to the 'incident_severity == Major Damage' rule. 1.0 means the model adds no "
                 "decision-level information beyond that rule (it can still add ranking)."),
    }

    champion_pipeline = fitted[champion_name]
    champion_threshold = chosen[champion_name]["threshold"]

    # ---- 5. Champion decision record — every number computed, none typed ----
    s_idx = summary.set_index("model")
    def _p(vs, metric):
        r = tests_df[(tests_df["vs"] == vs) & (tests_df["metric"] == metric)]
        return None if r.empty else {k: (float(r.iloc[0][k]) if k != "significant_at_0_05" else bool(r.iloc[0][k]))
                                     for k in ("mean_difference", "t", "p_corrected", "significant_at_0_05")}
    champion_decision = {
        "protocol": {
            "data": f"training split only ({len(y_train)} rows); the {len(y_test)}-row test split is never used for selection",
            "outer_cv": f"{ms.OUTER_SPLITS}-fold stratified x {ms.OUTER_REPEATS} repeats = {ms.OUTER_SPLITS * ms.OUTER_REPEATS} paired folds",
            "inner_cv": f"{ms.INNER_SPLITS}-fold, picks hyperparameters (PR-AUC) and threshold (F1) per candidate",
            "selection_rule": ("highest mean outer-fold F1 at each candidate's own tuned threshold; must not be "
                               f"worse than the '{ms.BASELINE_NAME}' baseline; ties within {ms.F1_TIE_TOLERANCE} F1 "
                               "broken by PR-AUC"),
            "significance_test": "Nadeau-Bengio corrected resampled t-test (two-sided) on the paired fold differences",
        },
        "measured_champion": champion_name,
        **decision,
        "champion_cv": {k: float(s_idx.loc[champion_name, k]) for k in s_idx.columns if k.endswith("_mean")},
        "baseline_cv": {k: float(s_idx.loc[ms.BASELINE_NAME, k]) for k in s_idx.columns if k.endswith("_mean")},
        "champion_vs_baseline": {m: _p(ms.BASELINE_NAME, m) for m in ("f1", "recall", "pr_auc", "roc_auc")},
        "champion_vs_others": {other: {m: _p(other, m) for m in ("f1", "recall", "pr_auc", "roc_auc")}
                               for other in fitted if other not in (champion_name, ms.BASELINE_NAME)},
        "pairwise_tests_csv": "data/processed/champion_pairwise_tests.csv",
    }
    with open(PROCESSED_DIR / "champion_decision.json", "w") as f:
        json.dump(champion_decision, f, indent=2)

    # ---- 6. Cost-threshold sensitivity (diagnostic only, train OOF) ----
    champ_factory = lambda: candidates[champion_name].factory(**chosen[champion_name]["params"])
    oof_train = _out_of_fold_proba(champ_factory, X_train, y_train)
    cost_result = find_cost_optimal_threshold(y_train.values, oof_train, train_df["total_claim_amount"].values, steps=50)
    cost_result["sweep"].to_csv(PROCESSED_DIR / "cost_threshold_sweep.csv", index=False)

    # ---- 7. Proxy-feature ablation (now train-only CV as well) ----
    proxy_ablation_df = run_proxy_feature_ablation(train_df, test_df, y_train, y_test, train_df, y_train)
    proxy_ablation_df.to_csv(PROCESSED_DIR / "proxy_feature_ablation.csv", index=False)

    # ---- 8. SHAP global importance (on training data the champion was fit on) ----
    background = X_train_scaled.sample(min(100, len(X_train_scaled)), random_state=RANDOM_STATE)
    explainer = ClaimExplainer(_final_estimator(champion_pipeline), feature_columns, background_data=background)
    shap_importance = explainer.global_importance(X_train_scaled)
    shap_importance["share_of_total"] = shap_importance["mean_abs_shap"] / shap_importance["mean_abs_shap"].sum()
    shap_importance.to_csv(PROCESSED_DIR / "shap_feature_importance.csv", index=False)
    # Same importance, grouped back to the ORIGINAL claim fields (one-hot
    # dummies summed) — what the scoring form's coverage check reads.
    source_of = build_source_map(feature_columns)
    shap_importance["source_field"] = shap_importance["feature"].map(lambda f: source_of.get(f, (f, None))[0])
    by_field = (shap_importance.groupby("source_field")[["mean_abs_shap", "share_of_total"]].sum()
                .sort_values("share_of_total", ascending=False).reset_index())
    by_field.to_csv(PROCESSED_DIR / "shap_importance_by_field.csv", index=False)

    # ---- 9. Save artifacts ----
    joblib.dump(champion_pipeline, MODELS_DIR / "champion_model.pkl")
    for name, fname in (("random_forest", "random_forest_final.pkl"),
                        ("logistic_regression", "logistic_regression_final.pkl"),
                        ("xgboost", "xgboost_final.pkl")):
        joblib.dump(fitted[name], MODELS_DIR / fname)
    joblib.dump(scaler, MODELS_DIR / "standard_scaler.pkl")
    background.to_csv(MODELS_DIR / "shap_background.csv", index=False)
    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_columns, f)

    metrics = {
        "primary_model": champion_name,
        "champion_artifact": "models/champion_model.pkl",
        "champion_hyperparameters": chosen[champion_name]["params"],
        "operating_threshold": champion_threshold,
        "model_comparison": comparison_rows,
        "baseline_agreement": baseline_agreement,
        "selection": {"summary": "data/processed/model_selection_summary.csv",
                      "decision": "data/processed/champion_decision.json"},
        "cost_optimal_threshold": {k: v for k, v in cost_result.items() if k != "sweep"},
        "n_train": len(X_train), "n_test": len(X_test),
        "n_features": len(feature_columns), "fraud_rate": float(y_train.mean()),
        "proxy_features": {
            "included_in_shipped_model": INCLUDE_PROXY_FEATURES,
            "risky_feature_columns": ["is_highrisk_hobby", "is_exec_occupation"],
            "ablation_report": "data/processed/proxy_feature_ablation.csv",
        },
        "uncertainty": {
            "cross_validation_mean_std": "data/processed/cross_validation_results.csv",
            "holdout_bootstrap_95ci": "data/processed/holdout_bootstrap_ci.csv",
        },
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    X_test_scaled.assign(y_true=y_test.values, y_proba=test_proba[champion_name]).to_csv(
        PROCESSED_DIR / "risk_scores_test.csv", index=False)
    X_test.assign(y_true=y_test.values).to_csv(PROCESSED_DIR / "psi_reference_features.csv", index=False)

    print()
    print("Nested CV on TRAIN (mean over folds):")
    print(summary.round(3).to_string(index=False))
    print()
    print(json.dumps({k: v for k, v in champion_decision.items() if k != "protocol"}, indent=2))
    print()
    print("Held-out TEST (reported once):")
    print(pd.DataFrame(comparison_rows)[["model", "threshold", "recall", "precision", "f1", "pr_auc", "roc_auc", "flag_rate"]].round(3).to_string(index=False))
    print()
    print("Agreement with Major-Damage rule:", json.dumps(baseline_agreement))


if __name__ == "__main__":
    main()
