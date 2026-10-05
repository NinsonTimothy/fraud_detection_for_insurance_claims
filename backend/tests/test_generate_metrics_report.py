"""test_generate_metrics_report.py — PB-15: this project's working rules
require every number quoted in docs/ to come from models/metrics.json (and
the other real data/processed/ artifacts), never be hand-typed. Before this
ticket that was enforced only by manual discipline — generate_metrics_report.py
is the dedicated script that actually generates a docs page from those
artifacts, and this file locks in that it produces internally-consistent,
correctly-sourced output rather than silently drifting or crashing on a
missing optional artifact."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.generate_metrics_report import build_report_markdown, load_champion_decision, load_metrics, load_oracle_report


def test_report_includes_the_real_shipped_model_and_threshold():
    metrics = load_metrics()
    report = build_report_markdown()
    assert metrics["primary_model"] in report
    primary_row = next(r for r in metrics["model_comparison"] if r["model"] == metrics["primary_model"])
    assert str(primary_row["threshold"]) in report


def test_report_model_comparison_table_matches_metrics_json_exactly():
    """Every recall/precision/f1/roc_auc figure in the generated table must
    be traceable byte-for-byte back to models/metrics.json — the whole
    point of this script existing."""
    metrics = load_metrics()
    report = build_report_markdown()
    for row in metrics["model_comparison"]:
        assert f"{row['roc_auc']:.3f}" in report
        assert f"{row['pr_auc']:.3f}" in report
        assert f"{row['f1']:.3f}" in report


def test_report_flags_when_oracle_ci_excludes_random_chance():
    """PB-15/SH-05: a concrete, reproduced instance of doc drift was found
    building this ticket — REBUILD_NOTES.md's own comparison table
    (written against an earlier training run) called the Oracle result
    "~0.48 (random)", but the CURRENT models/metrics.json-backed Oracle
    bootstrap CI no longer contains 0.5 at all, so "random" is no longer
    the most accurate characterization. This test locks in that the
    generator itself gets this distinction right, so it can never silently
    repeat that mistake."""
    oracle = load_oracle_report()
    if oracle is None or "oracle_metrics_ci" not in oracle:
        pytest.skip("no Oracle validation report on disk in this environment")
    report = build_report_markdown()
    roc_ci = oracle["oracle_metrics_ci"]["roc_auc"]
    contains_half = roc_ci["ci_lower"] <= 0.5 <= roc_ci["ci_upper"]
    # OR-01: wording is derived from the CI. Below 0.5 entirely => the
    # report must say "significantly inverted", never "random".
    if roc_ci["ci_upper"] < 0.5:
        assert "significantly inverted" in report
    elif contains_half:
        assert "indistinguishable from random" in report
    else:
        assert "better than random" in report


def test_report_includes_champion_selection_low_power_caveat_when_available():
    champion = load_champion_decision()
    if champion is None:
        pytest.skip("no champion_decision.json on disk in this environment")
    report = build_report_markdown()
    assert champion["measured_champion"] in report
    assert "corrected resampled t-test" in report


def test_report_p_values_come_from_the_computed_tests_csv_not_typed_text():
    """MS-01: the old champion note hardcoded "recall p=0.0086" while the
    computed value was 0.498. Every p-value in the report must now equal the
    one in champion_pairwise_tests.csv, and the stale figure must be gone."""
    import pandas as pd
    from app.ml.generate_metrics_report import PROCESSED_DIR
    path = PROCESSED_DIR / "champion_pairwise_tests.csv"
    if not path.exists():
        pytest.skip("no champion_pairwise_tests.csv on disk")
    report = build_report_markdown()
    assert "0.0086" not in report
    for _, r in pd.read_csv(path).iterrows():
        assert f"{r['p_corrected']:.3f}" in report


def test_report_is_written_to_docs_current_metrics(tmp_path, monkeypatch):
    """The script's main() writes docs/CURRENT_METRICS.md — verify the
    write actually lands and the file's content matches what
    build_report_markdown() returns, so `python -m
    app.ml.generate_metrics_report` isn't silently a no-op."""
    import app.ml.generate_metrics_report as gmr

    fake_docs_dir = tmp_path / "docs"
    fake_docs_dir.mkdir()
    monkeypatch.setattr(gmr, "DOCS_DIR", fake_docs_dir)

    gmr.main()

    out_path = fake_docs_dir / "CURRENT_METRICS.md"
    assert out_path.exists()
    assert out_path.read_text() == build_report_markdown()
