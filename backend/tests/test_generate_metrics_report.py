"""D3: the generated report must agree with the artifacts it is built from."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.ml.generate_metrics_report import (PROCESSED_DIR, build_report_markdown, load_champion_decision,
                                            load_metrics, load_oracle_report)
from app.ml.reporting import roc_ci_verdict

pytestmark = pytest.mark.skipif(load_metrics() is None, reason="artifacts not generated")


def test_report_names_computed_champion_and_threshold():
    m, r = load_metrics(), build_report_markdown()
    assert f"`{m['primary_model']}`" in r
    assert f"{m['operating_threshold']:.2f}" in r


def test_report_test_table_matches_metrics_json():
    m, r = load_metrics(), build_report_markdown()
    for row in m["model_comparison"]:
        assert f"| {row['model']} | {row['threshold']:.3f} | {row['recall']:.3f} | {row['precision']:.3f} | {row['f1']:.3f} |" in r


def test_report_p_values_equal_the_computed_csv():
    r = build_report_markdown()
    assert "0.0086" not in r and "0.0166" not in r
    for _, row in pd.read_csv(PROCESSED_DIR / "pairwise_tests.csv").iterrows():
        assert f"{row['p_corrected']:.3f}" in r


def test_oracle_wording_is_derived_from_ci():
    o = load_oracle_report()
    if o is None:
        pytest.skip("no Oracle report")
    short, _ = roc_ci_verdict(o["oracle_metrics"]["roc_auc"], o["oracle_metrics_ci"]["roc_auc"])
    assert f"Verdict: {short}" in build_report_markdown()


def test_roc_ci_verdict_three_cases():
    assert roc_ci_verdict(0.46, {"ci_lower": 0.44, "ci_upper": 0.48})[0] == "significantly inverted ranking"
    assert roc_ci_verdict(0.50, {"ci_lower": 0.48, "ci_upper": 0.52})[0] == "indistinguishable from random"
    assert roc_ci_verdict(0.60, {"ci_lower": 0.55, "ci_upper": 0.65})[0].startswith("better than random")


def test_current_metrics_md_on_disk_is_up_to_date():
    """docs == artifacts: the committed CURRENT_METRICS.md equals a fresh render."""
    on_disk = (PROCESSED_DIR.parents[1] / "docs" / "CURRENT_METRICS.md").read_text()
    assert on_disk == build_report_markdown()
