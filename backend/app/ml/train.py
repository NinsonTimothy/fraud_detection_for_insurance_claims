"""
train.py — the full, leak-free training pipeline (pre-defence round 2).

STRICT SPLIT (B1): 1,000 rows -> 800 DEVELOPMENT + 200 TEST (stratified,
seed 42). Every decision below is made on development rows only, using
cross-validation / out-of-fold (OOF) predictions:

  1. model selection (rule baseline + LR / RF / XGB; model_selection.py)
  2. proxy-feature ablation
  3. calibration decision (calibration.py)
  4. operational review threshold (F1-optimal on OOF scores)
  5. display risk bands (risk_policy.derive_band_edges)
  6. cost sensitivity grid (cost_threshold.py)

The 200 test rows are touched ONCE, in step 7, to REPORT. The row ids used
by each development step are written to data/processed/dev_phase_row_ids.json
and checked against the test ids by backend/tests (D3).

Nothing in this file states a result: every number, the champion's name and
every p-value are computed and written to artifacts.

Run: python -m app.ml.train   (from backend/) — or python -m app.ml.run_all
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from app.core.config import (HIGH_BAND_FRAUD_CAPTURE, INCLUDE_PROXY_FEATURES, MEDIUM_BAND_FRAUD_CAPTURE)
from app.ml import calibration as cal
from app.ml import model_selection as ms
from app.ml.cost_threshold import sensitivity_grid
from app.ml.explainer import ClaimExplainer
from app.ml.feature_engineering import engineer_features
from app.ml.inference import underlying_estimator
from app.ml.risk_policy import derive_band_edges
from app.ml.uncertainty import bootstrap_metric_ci

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_CLEANED = PROJECT_ROOT / "data" / "cleaned" / "insurance_claims_cleaned.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RANDOM_STATE = 42


# ------------------------------------------------------------- data split ---
def _read_cleaned() -> pd.DataFrame:
    # SH-01: keep authorities_contacted's genuine "None" category.
    return pd.read_csv(DATA_CLEANED, keep_default_na=False, na_values=[""])


def split_with_ids():
    """Development/test split that KEEPS the original row ids as the index."""
    df = _read_cleaned()
    y = (df["fraud_reported"] == "Y").astype(int)
    dev_df, test_df, y_dev, y_test = train_test_split(df, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    return dev_df, test_df, y_dev, y_test


def load_and_split():
    """Backward-compatible form (index reset) used by tests and experiments."""
    dev_df, test_df, y_dev, y_test = split_with_ids()
    return (dev_df.reset_index(drop=True), test_df.reset_index(drop=True),
            y_dev.reset_index(drop=True), y_test.reset_index(drop=True))


def build_features(train_df, test_df, y_train=None, include_proxy_features: bool | None = None):
    X_train = engineer_features(train_df, include_proxy_features=include_proxy_features)
    X_test = engineer_features(test_df, include_proxy_features=include_proxy_features)
    return X_train, X_test.reindex(columns=X_train.columns, fill_value=0)


# --------------------------------------------- helpers kept for the tests ---
def _out_of_fold_proba(make_estimator, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> np.ndarray:
    return ms.oof_proba(make_estimator, X.to_numpy(dtype=float), np.asarray(y), n_splits=n_splits)


def _f1_optimal_threshold(y_true, proba, grid) -> float:
    return ms.f1_optimal_threshold(y_true, proba, grid)


def cross_validate_model(name, make_estimator, full_df: pd.DataFrame, y_all: pd.Series, include_proxy_features=None):
    """Full-pipeline 5-fold CV (features, scaler and model refit per fold)."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    out = {k: [] for k in ("roc_auc", "pr_auc", "f1", "recall", "precision")}
    for tr, va in skf.split(full_df, y_all):
        X_tr = engineer_features(full_df.iloc[tr], include_proxy_features=include_proxy_features)
        X_va = engineer_features(full_df.iloc[va], include_proxy_features=include_proxy_features).reindex(columns=X_tr.columns, fill_value=0)
        p = ms.fit_predict(make_estimator, X_tr.to_numpy(float), y_all.iloc[tr].to_numpy(), X_va.to_numpy(float))
        m = ms.classification_metrics(y_all.iloc[va], p, 0.5)
        out["roc_auc"].append(roc_auc_score(y_all.iloc[va], p)); out["pr_auc"].append(average_precision_score(y_all.iloc[va], p))
        for k in ("f1", "recall", "precision"):
            out[k].append(m[k])
    return {"model": name, **{f"{k}_mean": float(np.mean(v)) for k, v in out.items()},
            **{f"{k}_std": float(np.std(v)) for k, v in out.items()}}


def evaluate(name, model, X_test, y_test, threshold=0.5) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    m = ms.classification_metrics(y_test, proba, threshold)
    return {"model": name, "threshold": float(threshold), "recall": m["recall"], "precision": m["precision"],
            "f1": m["f1"], "pr_auc": float(average_precision_score(y_test, proba)),
            "roc_auc": float(roc_auc_score(y_test, proba)), "accuracy": m["accuracy"],
            "flag_rate": m["flag_rate"], "confusion_matrix": m["confusion_matrix"]}


# ------------------------------------------------------ development steps ---
def proxy_ablation_dev(dev_df, y_dev, make_estimator) -> pd.DataFrame:
    """B1: proxy features ON vs OFF, champion family, development rows only,
    repeated 5x2 CV, threshold-free metrics + corrected t-test."""
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE)
    rows = []
    for fold, (tr, va) in enumerate(rskf.split(dev_df, y_dev)):
        for include in (False, True):
            X_tr = engineer_features(dev_df.iloc[tr], include_proxy_features=include)
            X_va = engineer_features(dev_df.iloc[va], include_proxy_features=include).reindex(columns=X_tr.columns, fill_value=0)
            p = ms.fit_predict(make_estimator, X_tr.to_numpy(float), y_dev.iloc[tr].to_numpy(), X_va.to_numpy(float))
            rows.append({"fold": fold, "variant": "proxy_features_on" if include else "proxy_features_off",
                         "n_features": X_tr.shape[1], "pr_auc": average_precision_score(y_dev.iloc[va], p),
                         "roc_auc": roc_auc_score(y_dev.iloc[va], p), "n_train": len(tr), "n_val": len(va)})
    f = pd.DataFrame(rows)
    summary = f.groupby("variant").agg(n_features=("n_features", "first"), pr_auc_mean=("pr_auc", "mean"),
                                       pr_auc_std=("pr_auc", "std"), roc_auc_mean=("roc_auc", "mean"),
                                       roc_auc_std=("roc_auc", "std")).reset_index()
    wide = f.pivot(index="fold", columns="variant", values="pr_auc")
    t, p = ms.corrected_resampled_ttest((wide["proxy_features_on"] - wide["proxy_features_off"]).to_numpy(),
                                        int(f["n_train"].iloc[0]), int(f["n_val"].iloc[0]))
    summary["pr_auc_on_minus_off_p_corrected"] = p
    summary["shipped"] = summary["variant"].eq("proxy_features_on" if INCLUDE_PROXY_FEATURES else "proxy_features_off")
    return summary


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    dev_df, test_df, y_dev, y_test = split_with_ids()
    dev_ids, test_ids = sorted(map(int, dev_df.index)), sorted(map(int, test_df.index))
    X_dev, X_test = build_features(dev_df, test_df)
    cols = list(X_dev.columns)
    Xd, yd = X_dev.to_numpy(float), y_dev.to_numpy()
    used = {}

    # 1. model selection (dev only)
    print(f"[1/7] model selection: {ms.CV_SPLITS}x{ms.CV_REPEATS} repeated CV on {len(yd)} development rows ...", flush=True)
    sel = ms.nested_cv_compare(X_dev, y_dev, cols)
    used["model_selection"] = sel["row_ids_seen"]
    folds, cands = sel["folds"], {c.name: c for c in sel["candidates"]}
    folds.to_csv(PROCESSED_DIR / "model_selection_folds.csv", index=False)
    summary = ms.summarize(folds)
    summary.to_csv(PROCESSED_DIR / "model_selection_summary.csv", index=False)
    summary[["model"] + [c for c in summary.columns if c.endswith(("_mean", "_std"))]].to_csv(
        PROCESSED_DIR / "cross_validation_results.csv", index=False)
    tests = ms.all_pairwise_tests(folds)
    tests.to_csv(PROCESSED_DIR / "pairwise_tests.csv", index=False)
    decision = ms.select_champion(folds, sel["candidates"])
    champ = decision["champion"]
    print(f"      champion (computed): {champ}", flush=True)

    # 2. proxy ablation (dev only)
    print("[2/7] proxy-feature ablation (dev) ...", flush=True)
    proxy_ablation_dev(dev_df, y_dev, cands[champ].factory).to_csv(PROCESSED_DIR / "proxy_feature_ablation.csv", index=False)
    used["proxy_ablation"] = dev_ids

    # 3. calibration (dev only) + per-model dev reliability
    print("[3/7] calibration decision (dev) ...", flush=True)
    oof_fn = lambda factory, X, y: ms.oof_proba(factory, X, y, seed=RANDOM_STATE)
    calib = cal.choose_calibration(cands[champ].factory, Xd, yd, oof_fn)
    used["calibration"] = dev_ids
    champ_factory = cal.make_calibrated_factory(cands[champ].factory, None if calib["chosen"] == "none" else calib["chosen"])
    champ_oof = calib["_oof_chosen"]
    dev_oof = {n: (champ_oof if n == champ else oof_fn(c.factory, Xd, yd)) for n, c in cands.items()}

    # 4-6. threshold, bands, cost grid from the champion's dev OOF scores
    print("[4/7] operating threshold, bands and cost sensitivity (dev OOF) ...", flush=True)
    operating_threshold = ms.f1_optimal_threshold(yd, champ_oof)
    edges = derive_band_edges(yd, champ_oof, HIGH_BAND_FRAUD_CAPTURE, MEDIUM_BAND_FRAUD_CAPTURE)
    cost_best, cost_sweep = sensitivity_grid(yd, champ_oof, dev_df["total_claim_amount"].to_numpy())
    cost_best.to_csv(PROCESSED_DIR / "cost_sensitivity.csv", index=False)
    cost_sweep.to_csv(PROCESSED_DIR / "cost_threshold_sweep.csv", index=False)
    used["threshold_bands_cost"] = dev_ids
    dev_thresholds = {n: (0.5 if c.is_baseline else (operating_threshold if n == champ else ms.f1_optimal_threshold(yd, dev_oof[n])))
                      for n, c in cands.items()}

    # final fits on ALL development rows
    scaler = StandardScaler().fit(X_dev)
    Xd_s = pd.DataFrame(scaler.transform(X_dev), columns=cols, index=X_dev.index)
    Xt_s = pd.DataFrame(scaler.transform(X_test), columns=cols, index=X_test.index)
    fitted = {}
    for n, c in cands.items():
        fitted[n] = (champ_factory if n == champ else c.factory)()
        fitted[n].fit(Xd_s, yd)

    # 7. THE ONE LOOK AT THE TEST SET — reporting only
    print("[5/7] final evaluation on the 200 test rows (first and only use) ...", flush=True)
    yt = y_test.to_numpy()
    rows, test_proba, rel_rows, calib_rows, boot_rows = [], {}, [], [], []
    for n in [champ] + [m for m in cands if m not in (champ, ms.BASELINE_NAME)] + [ms.BASELINE_NAME]:
        p = fitted[n].predict_proba(Xt_s)[:, 1]
        test_proba[n] = p
        r = evaluate(n, fitted[n], Xt_s, yt, dev_thresholds[n])
        r.update(is_champion=n == champ, is_baseline=cands[n].is_baseline, **{f"test_{k}": v for k, v in cal.calibration_summary(yt, p).items()})
        rows.append(r)
        for split, yy, pp in (("development_oof", yd, dev_oof[n]), ("test", yt, p)):
            t = cal.reliability_table(yy, pp); t.insert(0, "split", split); t.insert(0, "model", n); rel_rows.append(t)
            calib_rows.append({"model": n, "split": split, **cal.calibration_summary(yy, pp)})
        for metric, iv in bootstrap_metric_ci(yt, p, threshold=dev_thresholds[n], n_boot=1000, random_state=RANDOM_STATE).items():
            boot_rows.append({"model": n, "metric": metric, "point_estimate": r[metric], **iv})
    pd.DataFrame(rows).drop(columns=["confusion_matrix"]).to_csv(PROCESSED_DIR / "model_comparison.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(PROCESSED_DIR / "holdout_bootstrap_ci.csv", index=False)
    pd.concat(rel_rows).to_csv(PROCESSED_DIR / "reliability_curves.csv", index=False)
    pd.DataFrame(calib_rows).to_csv(PROCESSED_DIR / "calibration_summary.csv", index=False)
    _reliability_png(pd.concat(rel_rows), champ)

    flag_c = test_proba[champ] >= operating_threshold
    flag_r = test_proba[ms.BASELINE_NAME] >= 0.5
    agreement = {"test_decision_agreement": float((flag_c == flag_r).mean()),
                 "flagged_by_champion_not_rule": int((flag_c & ~flag_r).sum()),
                 "flagged_by_rule_not_champion": int((~flag_c & flag_r).sum()), "n_test": int(len(yt))}

    # SHAP (explains the champion's underlying estimator; dev rows)
    print("[6/7] SHAP importance (dev) ...", flush=True)
    background = Xd_s.sample(min(100, len(Xd_s)), random_state=RANDOM_STATE)
    expl = ClaimExplainer(underlying_estimator(fitted[champ]), cols, background_data=background)
    col_imp = expl.global_importance(Xd_s)
    col_imp["share_of_total"] = col_imp["mean_abs_shap"] / col_imp["mean_abs_shap"].sum()
    col_imp.to_csv(PROCESSED_DIR / "shap_feature_importance.csv", index=False)
    field_imp = expl.grouped_global_importance(Xd_s)
    field_imp.rename(columns={"field": "source_field"}).to_csv(PROCESSED_DIR / "shap_importance_by_field.csv", index=False)

    # save
    print("[7/7] saving artifacts ...", flush=True)
    joblib.dump(fitted[champ], MODELS_DIR / "champion_model.pkl")
    for n, fn in (("random_forest", "random_forest_final.pkl"), ("logistic_regression", "logistic_regression_final.pkl"),
                  ("xgboost", "xgboost_final.pkl")):
        joblib.dump(fitted[n], MODELS_DIR / fn)
    joblib.dump(scaler, MODELS_DIR / "standard_scaler.pkl")
    background.to_csv(MODELS_DIR / "shap_background.csv", index=False)
    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(cols, f)
    with open(MODELS_DIR / "risk_policy.json", "w") as f:
        json.dump(edges, f, indent=2)
    with open(PROCESSED_DIR / "dev_phase_row_ids.json", "w") as f:
        json.dump({"development_ids": dev_ids, "test_ids": test_ids, "steps": used}, f)

    decision_out = {
        "protocol": {
            "data": f"development split only ({len(yd)} rows); the {len(yt)}-row test split is used once, for reporting",
            "cv": f"repeated stratified {ms.CV_SPLITS}-fold x {ms.CV_REPEATS} repeats = {ms.CV_SPLITS * ms.CV_REPEATS} paired folds",
            "thresholds": "each ML candidate: F1-optimal threshold on inner 5-fold out-of-fold predictions; rule: fixed (binary output)",
            "imbalance": "class weighting only (no SMOTE)",
            "significance_test": f"Nadeau-Bengio corrected resampled t-test, two-sided, alpha={ms.SIGNIFICANCE_ALPHA}",
            "selection_rule": ("best mean PR-AUC among ML models, unless a simpler model (LR < RF < XGB) is not "
                               "significantly worse on PR-AUC, in which case the simplest such model; the rule "
                               "baseline is reported but not eligible"),
        },
        **decision,
        "champion_cv_means": {k: float(v) for k, v in summary.set_index("model").loc[champ].items() if k.endswith("_mean")},
        "rule_cv_means": {k: float(v) for k, v in summary.set_index("model").loc[ms.BASELINE_NAME].items() if k.endswith("_mean")},
    }
    with open(PROCESSED_DIR / "champion_decision.json", "w") as f:
        json.dump(decision_out, f, indent=2, default=float)

    metrics = {
        "primary_model": champ, "champion_artifact": "models/champion_model.pkl",
        "operating_threshold": operating_threshold,
        "risk_bands": {k: edges[k] for k in ("medium_edge", "high_edge")},
        "calibration": {k: v for k, v in calib.items() if not k.startswith("_")},
        "model_comparison": rows, "baseline_agreement": agreement,
        "n_train": len(yd), "n_test": len(yt), "n_features": len(cols), "fraud_rate": float(yd.mean()),
        "proxy_features": {"included_in_shipped_model": INCLUDE_PROXY_FEATURES,
                           "risky_feature_columns": ["is_highrisk_hobby", "is_exec_occupation"],
                           "ablation_report": "data/processed/proxy_feature_ablation.csv"},
        "uncertainty": {"cross_validation_mean_std": "data/processed/cross_validation_results.csv",
                        "holdout_bootstrap_95ci": "data/processed/holdout_bootstrap_ci.csv"},
        "artifacts": {"selection": "data/processed/model_selection_summary.csv",
                      "pairwise_tests": "data/processed/pairwise_tests.csv",
                      "decision": "data/processed/champion_decision.json",
                      "calibration": "data/processed/calibration_summary.csv",
                      "cost_sensitivity": "data/processed/cost_sensitivity.csv",
                      "bootstrap_ci": "data/processed/holdout_bootstrap_ci.csv"},
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)

    Xt_s.assign(y_true=yt, y_proba=test_proba[champ]).to_csv(PROCESSED_DIR / "risk_scores_test.csv", index=False)
    X_test.assign(y_true=yt).to_csv(PROCESSED_DIR / "psi_reference_features.csv", index=False)
    print(summary.round(3).to_string(index=False))
    print(json.dumps({"champion": champ, "calibration": calib["chosen"], "operating_threshold": operating_threshold,
                      "bands": {k: edges[k] for k in ("medium_edge", "high_edge")}, "agreement_with_rule": agreement}, indent=2))


def _reliability_png(rel: pd.DataFrame, champion: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, split in zip(axes, ("development_oof", "test")):
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        for model, g in rel[(rel["split"] == split) & (rel["n"] > 0)].groupby("model"):
            ax.plot(g["mean_predicted"], g["observed_fraud_rate"], "o-", lw=2.5 if model == champion else 1,
                    label=f"{model}{' (champion)' if model == champion else ''}")
        ax.set_title(f"Reliability — {split}"); ax.set_xlabel("mean predicted score"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    axes[0].set_ylabel("observed fraud rate"); axes[1].legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(PROCESSED_DIR / "reliability_plot.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
