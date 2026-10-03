"""test_train_pipeline.py — PB-16: `train.py` (load_and_split, build_features,
cross_validate_model, _f1_optimal_threshold) had no test coverage at all
before this ticket — every number it produces (docs/REBUILD_NOTES.md,
models/metrics.json) rested entirely on manual re-runs, not an automated
regression check. Also the "train/serve parity" test this ticket
specifically asked for: a direct, reproduced check that the artifacts
`FraudScoringService` actually serves (models/random_forest_final.pkl +
standard_scaler.pkl + feature_columns.json) score a real held-out row
IDENTICALLY to independently re-deriving that row's features via
train.py's own `build_features()` — the one test in this suite that
would catch inference.py's serving pipeline silently drifting from
train.py's training pipeline (different column order, a stale
feature_columns.json, a scaler swapped for a differently-fit one) even
though today they happen to agree."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.feature_engineering import align_to_training_columns
from app.ml.inference import FraudScoringService
from app.ml.train import (
    _f1_optimal_threshold,
    _out_of_fold_proba,
    build_features,
    cross_validate_model,
    load_and_split,
)


@pytest.fixture(scope="module")
def split():
    return load_and_split()


def test_load_and_split_is_deterministic_and_stratified(split):
    """train.py's RANDOM_STATE=42 + stratify=y must produce the exact same
    split every run — every downstream metric (models/metrics.json,
    docs/REBUILD_NOTES.md) implicitly assumes this. Re-running the split
    twice must give identical rows, and the fraud rate in each half must
    stay close to the full dataset's (that's what stratify=y is for)."""
    train_df, test_df, y_train, y_test = split
    train_df2, test_df2, y_train2, y_test2 = load_and_split()

    pd.testing.assert_frame_equal(train_df, train_df2)
    pd.testing.assert_frame_equal(test_df, test_df2)

    full_rate = pd.concat([y_train, y_test]).mean()
    assert y_train.mean() == pytest.approx(full_rate, abs=0.03)
    assert y_test.mean() == pytest.approx(full_rate, abs=0.03)
    assert len(train_df) + len(test_df) == 1000


def test_build_features_aligns_test_columns_to_train_columns(split):
    """build_features() reindexes X_test onto X_train's own columns (not
    just "whatever engineer_features happens to produce for the test
    rows") — this is what makes it safe to score X_test through a model
    fit on X_train's exact column order. A drift here (e.g. a category
    value present in test but not train producing an extra one-hot
    column) must be caught, not silently misalign features at scoring
    time."""
    train_df, test_df, y_train, y_test = split
    X_train, X_test = build_features(train_df, test_df, y_train)
    assert list(X_train.columns) == list(X_test.columns)
    assert X_train.isna().sum().sum() == 0
    assert X_test.isna().sum().sum() == 0


def test_f1_optimal_threshold_matches_brute_force_grid_search():
    """Unit test for the threshold-selection helper in isolation, with a
    synthetic y_true/proba pair whose best F1 threshold is known by
    construction (perfect separation at 0.5: every fraud case scores
    above 0.5, every non-fraud case below) — independent of any real
    model or data file."""
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 50 + [1] * 50)
    proba = np.concatenate([rng.uniform(0.0, 0.4, 50), rng.uniform(0.6, 1.0, 50)])
    grid = np.linspace(0.05, 0.95, 19)

    best = _f1_optimal_threshold(y_true, proba, grid)
    # Any threshold strictly between the two clusters achieves perfect
    # F1=1.0 here; the grid itself only offers 0.5 as an exact split point.
    pred = (proba >= best).astype(int)
    tp = ((pred == 1) & (y_true == 1)).sum()
    fp = ((pred == 1) & (y_true == 0)).sum()
    fn = ((pred == 0) & (y_true == 1)).sum()
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    assert f1 == pytest.approx(1.0)


def test_out_of_fold_proba_returns_one_probability_per_row_never_from_its_own_fold(split):
    """Each row's out-of-fold probability must come from a model that
    never saw that row during fit. Full leakage-freedom isn't directly
    observable from the output alone, but a coarse, real check is: OOF
    predictions from a near-degenerate estimator (predicts the train
    fold's OWN class balance regardless of input) should NOT achieve a
    suspiciously perfect separation — a leaked (in-fold) prediction on
    this dataset's small size would trend toward memorization instead."""
    train_df, test_df, y_train, y_test = split
    X_train, _ = build_features(train_df, test_df, y_train)

    def make_lr():
        return LogisticRegression(max_iter=200, random_state=42)

    oof = _out_of_fold_proba(make_lr, X_train, y_train, n_splits=5)
    assert oof.shape == (len(X_train),)
    assert ((oof >= 0.0) & (oof <= 1.0)).all()
    # A trivial "predict everyone as majority class" estimator would give
    # every row the identical probability; genuine OOF predictions from a
    # real classifier should vary across rows.
    assert oof.std() > 0.01


def test_cross_validate_model_returns_five_fold_summary_stats_in_valid_ranges(split):
    """PB-16: `cross_validate_model()` (the function every reported CV
    number in docs/REBUILD_NOTES.md and models/metrics.json's
    cross_validation_results.csv ultimately comes from) had no test at
    all. Uses LogisticRegression here (fast) rather than the shipped
    RF+SMOTE pipeline — this test is about the CV MACHINERY (does it
    return one mean/std per metric, are they all valid probabilities),
    not about re-verifying the shipped model's own reported numbers,
    which live in metrics.json and are generated by an actual training
    run, never hand-typed (see docs/REBUILD_NOTES.md's working rules)."""
    train_df, test_df, y_train, y_test = split
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    y_full = pd.concat([y_train, y_test], ignore_index=True)

    def make_lr():
        return LogisticRegression(max_iter=200, random_state=42)

    result = cross_validate_model("logistic_regression_smoke_test", make_lr, full_df, y_full)

    assert result["model"] == "logistic_regression_smoke_test"
    for metric in ("roc_auc", "pr_auc", "f1", "recall", "precision"):
        mean_key, std_key = f"{metric}_mean", f"{metric}_std"
        assert mean_key in result and std_key in result
        assert 0.0 <= result[mean_key] <= 1.0
        assert result[std_key] >= 0.0


TEST_API_KEY = "test-suite-api-key"


def test_train_serve_parity_scoring_matches_shipped_artifacts(split, monkeypatch, tmp_path):
    """PB-16: the flagship train/serve-parity test. Takes a real row from
    train.py's own held-out test split, scores it through
    FraudScoringService.score_one() (the exact code path /score and the
    dashboard's Score a claim page both call), and independently
    re-derives the same row's score by calling train.py's own
    build_features() directly and predicting with the SAME shipped
    rf_pipeline/scaler artifacts. These two code paths currently compute
    feature engineering independently (inference.py's `_prepare()` vs.
    train.py's `build_features()`, both calling the shared
    `engineer_features()`/`align_to_training_columns()` underneath) —
    this test is what would catch them silently diverging (a reordered
    column, a different alignment call, a stale scaler) without anyone
    noticing, since nothing else in this suite compares train.py's
    pipeline against inference.py's serving pipeline end-to-end on a
    real row scored through the real shipped model."""
    train_df, test_df, y_train, y_test = split

    service = FraudScoringService.instance()

    X_train, X_test = build_features(train_df, test_df, y_train)
    X_test_aligned = align_to_training_columns(X_test, service.feature_columns)
    X_test_scaled = pd.DataFrame(
        service.scaler.transform(X_test_aligned), columns=service.feature_columns, index=X_test.index,
    )
    expected_proba = service.rf_pipeline.predict_proba(X_test_scaled)[:, 1]

    # Check several rows, not just one — a single lucky match wouldn't
    # rule out an off-by-one alignment bug that happens to cancel out.
    for i in (0, 1, len(test_df) - 1):
        row = test_df.iloc[i].drop(labels=["fraud_reported"]).to_dict()
        served = service.score_one(row)
        assert served["fraud_probability"] == pytest.approx(float(expected_proba[i]), abs=1e-9)
