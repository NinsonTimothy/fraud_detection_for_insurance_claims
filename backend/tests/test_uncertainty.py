"""test_uncertainty.py — regression coverage for SH-04: bootstrap 95% CI
for point-estimate metrics, reported alongside cross_validate_model()'s
existing cross-fold mean±SD (a different, complementary source of
uncertainty — see uncertainty.py's module docstring)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.uncertainty import bootstrap_metric_ci

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _synthetic_labels_and_proba(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.25).astype(int)
    # Proba correlated with y but noisy — a realistic imperfect classifier.
    proba = np.clip(y * 0.5 + rng.normal(0, 0.25, size=n) + 0.25, 0, 1)
    return y, proba


def test_bootstrap_ci_contains_point_estimate_for_roc_auc_and_pr_auc():
    from sklearn.metrics import average_precision_score, roc_auc_score
    y, proba = _synthetic_labels_and_proba()
    point_roc = roc_auc_score(y, proba)
    point_pr = average_precision_score(y, proba)
    ci = bootstrap_metric_ci(y, proba, threshold=0.5, n_boot=500, random_state=42)
    assert ci["roc_auc"]["ci_lower"] <= point_roc <= ci["roc_auc"]["ci_upper"]
    assert ci["pr_auc"]["ci_lower"] <= point_pr <= ci["pr_auc"]["ci_upper"]


def test_bootstrap_ci_without_threshold_only_returns_auc_metrics():
    y, proba = _synthetic_labels_and_proba()
    ci = bootstrap_metric_ci(y, proba, threshold=None, n_boot=200, random_state=42)
    assert set(ci.keys()) == {"roc_auc", "pr_auc"}


def test_bootstrap_ci_with_threshold_returns_classification_metrics_too():
    y, proba = _synthetic_labels_and_proba()
    ci = bootstrap_metric_ci(y, proba, threshold=0.5, n_boot=200, random_state=42)
    assert {"roc_auc", "pr_auc", "recall", "precision", "f1", "accuracy"} <= set(ci.keys())
    for metric, interval in ci.items():
        assert interval["ci_lower"] <= interval["ci_upper"]
        assert interval["n_boot_effective"] > 0


def test_bootstrap_ci_is_deterministic_given_random_state():
    y, proba = _synthetic_labels_and_proba()
    ci1 = bootstrap_metric_ci(y, proba, threshold=0.5, n_boot=300, random_state=7)
    ci2 = bootstrap_metric_ci(y, proba, threshold=0.5, n_boot=300, random_state=7)
    assert ci1 == ci2


def test_bootstrap_ci_empty_input_returns_empty_dict():
    assert bootstrap_metric_ci([], [], threshold=0.5) == {}


def test_bootstrap_ci_degenerate_all_one_class_handles_auc_gracefully():
    """A degenerate all-negative sample can't score ROC-AUC/PR-AUC on ANY
    resample (every resample is also all-negative) — the metric should
    simply be absent from the result (n_boot_effective=0 case), not crash."""
    y = np.zeros(50, dtype=int)
    proba = np.random.default_rng(0).random(50)
    ci = bootstrap_metric_ci(y, proba, threshold=0.5, n_boot=100, random_state=1)
    assert "roc_auc" not in ci
    assert "pr_auc" not in ci
    # recall/precision/f1/accuracy ARE still well-defined for an all-negative sample.
    assert "accuracy" in ci


def test_holdout_bootstrap_ci_artifact_matches_model_comparison():
    import pandas as pd
    bootstrap_path = PROJECT_ROOT / "data" / "processed" / "holdout_bootstrap_ci.csv"
    comparison_path = PROJECT_ROOT / "data" / "processed" / "model_comparison.csv"
    if not (bootstrap_path.exists() and comparison_path.exists()):
        pytest.skip("artifacts not generated yet — run `python -m app.ml.train` first")
    bootstrap_df = pd.read_csv(bootstrap_path)
    comparison_df = pd.read_csv(comparison_path)
    assert set(bootstrap_df["model"]) == set(comparison_df["model"])
    for _, row in bootstrap_df.iterrows():
        assert row["ci_lower"] <= row["point_estimate"] <= row["ci_upper"] + 1e-9


def test_metrics_json_discloses_uncertainty_artifacts():
    import json
    metrics_path = PROJECT_ROOT / "models" / "metrics.json"
    if not metrics_path.exists():
        pytest.skip("models/metrics.json not generated yet — run `python -m app.ml.train` first")
    with open(metrics_path) as f:
        metrics = json.load(f)
    assert "uncertainty" in metrics
    assert "cross_validation_mean_std" in metrics["uncertainty"]
    assert "holdout_bootstrap_95ci" in metrics["uncertainty"]
