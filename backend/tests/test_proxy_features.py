"""test_proxy_features.py — regression coverage for SH-02 / D3: gate
`is_highrisk_hobby`/`is_exec_occupation` (feature_engineering.RISKY_FEATURE_COLUMNS)
behind `INCLUDE_PROXY_FEATURES` (app.core.config, default False — the
deployable headline model excludes them; see that module's own docstring
for why they're flagged as close to proxy-discrimination, not genuine
fraud signal)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import INCLUDE_PROXY_FEATURES
from app.ml.feature_engineering import RISKY_FEATURE_COLUMNS, engineer_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def sample_df():
    df = pd.read_csv(
        PROJECT_ROOT / "data" / "cleaned" / "insurance_claims_cleaned.csv",
        keep_default_na=False, na_values=[""],
    )
    return df.iloc[:20].reset_index(drop=True)


def test_config_default_is_off():
    """SH-02 / D3's default is OFF — the deployable headline model must
    not include the proxy features unless explicitly opted in."""
    assert INCLUDE_PROXY_FEATURES is False


def test_engineer_features_respects_default_config(sample_df):
    X = engineer_features(sample_df)  # no explicit override -> reads config
    for col in RISKY_FEATURE_COLUMNS:
        assert col not in X.columns, f"{col} should be absent under the default (OFF) config"


def test_engineer_features_explicit_true_includes_proxy_columns(sample_df):
    X = engineer_features(sample_df, include_proxy_features=True)
    for col in RISKY_FEATURE_COLUMNS:
        assert col in X.columns


def test_engineer_features_explicit_false_excludes_proxy_columns(sample_df):
    X = engineer_features(sample_df, include_proxy_features=False)
    for col in RISKY_FEATURE_COLUMNS:
        assert col not in X.columns


def test_explicit_true_produces_more_columns_than_false(sample_df):
    X_on = engineer_features(sample_df, include_proxy_features=True)
    X_off = engineer_features(sample_df, include_proxy_features=False)
    assert len(X_on.columns) == len(X_off.columns) + len(RISKY_FEATURE_COLUMNS)


def test_shipped_feature_columns_json_excludes_proxy_features_by_default():
    """The actually-shipped artifact must match the deployable (OFF)
    default — this is the "OFF is the deployable headline" requirement,
    not just something engineer_features() is theoretically capable of."""
    import json
    fc_path = PROJECT_ROOT / "models" / "feature_columns.json"
    if not fc_path.exists():
        pytest.skip("models/feature_columns.json not generated yet — run `python -m app.ml.train` first")
    with open(fc_path) as f:
        feature_columns = json.load(f)
    for col in RISKY_FEATURE_COLUMNS:
        assert col not in feature_columns


def test_metrics_json_discloses_proxy_feature_setting():
    import json
    metrics_path = PROJECT_ROOT / "models" / "metrics.json"
    if not metrics_path.exists():
        pytest.skip("models/metrics.json not generated yet — run `python -m app.ml.train` first")
    with open(metrics_path) as f:
        metrics = json.load(f)
    assert "proxy_features" in metrics
    assert metrics["proxy_features"]["included_in_shipped_model"] is False
    assert set(metrics["proxy_features"]["risky_feature_columns"]) == set(RISKY_FEATURE_COLUMNS)


def test_proxy_feature_ablation_report_has_both_variants():
    ablation_path = PROJECT_ROOT / "data" / "processed" / "proxy_feature_ablation.csv"
    if not ablation_path.exists():
        pytest.skip("proxy_feature_ablation.csv not generated yet — run `python -m app.ml.train` first")
    df = pd.read_csv(ablation_path)
    assert set(df["variant"]) == {"proxy_features_on", "proxy_features_off"}
    on_row = df[df["variant"] == "proxy_features_on"].iloc[0]
    off_row = df[df["variant"] == "proxy_features_off"].iloc[0]
    assert on_row["n_features"] == off_row["n_features"] + len(RISKY_FEATURE_COLUMNS)
    # Both variants must report real, finite metrics — not placeholders.
    for col in ("holdout_recall", "holdout_roc_auc", "cv_recall_mean", "cv_roc_auc_mean"):
        assert 0.0 <= on_row[col] <= 1.0
        assert 0.0 <= off_row[col] <= 1.0
