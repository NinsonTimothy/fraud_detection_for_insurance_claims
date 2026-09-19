"""dashboard/tests/test_dashboard_pages.py — regression coverage for PB-01
(dashboard sys.path shadowing crash).

Each Streamlit rerun re-executes a page's top-level code. The bug: every
page inserted its own directory at sys.path[0] on every rerun, so once
the dashboard's entry script was literally named app.py, a later
`from app.ml.inference import FraudScoringService` (inside
components.data_access.get_scoring_service, deferred behind
@st.cache_resource so it only actually imports on first call — which can
land on a LATER rerun than page load) resolved `app` to dashboard/app.py
instead of the backend's app/ package. Fixed by renaming the entry script
to streamlit_app.py and switching every page's sys.path insert from
insert(0, ...) to a guarded append(...), so backend/ (inserted once, at
position 0, by data_access.py) is never pushed behind the dashboard dir.

These tests run each page in a FRESH interpreter process (via pytest's
own collection, one file = one process for the whole session — good
enough here since @st.cache_resource state is process-global and the
bug was specifically about import resolution, not cross-test leakage)
and interact with real widgets, not just import the module.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

DASHBOARD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DASHBOARD_ROOT.parent


def test_score_claim_submit_renders_probability_no_exception():
    at = AppTest.from_file(str(DASHBOARD_ROOT / "app_pages" / "score_claim.py"))
    at.run(timeout=30)
    assert not at.exception, [str(e) for e in at.exception]

    at.button[0].click().run(timeout=30)
    assert not at.exception, [str(e) for e in at.exception]

    headings = [md.value for md in at.markdown if "fraud probability" in md.value]
    assert headings, "expected a '### NN.N% fraud probability' heading to render"


def test_batch_review_upload_scores_claims_no_exception():
    at = AppTest.from_file(str(DASHBOARD_ROOT / "app_pages" / "batch_review.py"))
    at.run(timeout=30)
    assert not at.exception, [str(e) for e in at.exception]

    raw = REPO_ROOT / "data" / "raw" / "insurance_claims_raw.csv"
    sample = pd.read_csv(raw).head(5)
    csv_bytes = sample.to_csv(index=False).encode()

    at.file_uploader[0].set_value(("batch_test.csv", csv_bytes, "text/csv")).run(timeout=30)
    assert not at.exception, [str(e) for e in at.exception]

    metrics = [m.value for m in at.metric]
    assert metrics and metrics[0] == "5", f"expected 'Claims scored' == 5, got {metrics}"


def test_overview_then_score_in_sequence_no_exception():
    """PB-01's acceptance criteria explicitly calls for running pages IN
    SEQUENCE (Overview -> Score) within one process, not just each in
    isolation — this is what actually exercises whether an earlier
    page's sys.path mutation corrupts a later page's imports."""
    overview = AppTest.from_file(str(DASHBOARD_ROOT / "app_pages" / "overview.py"))
    overview.run(timeout=30)
    assert not overview.exception, [str(e) for e in overview.exception]

    score = AppTest.from_file(str(DASHBOARD_ROOT / "app_pages" / "score_claim.py"))
    score.run(timeout=30)
    score.button[0].click().run(timeout=30)
    assert not score.exception, [str(e) for e in score.exception]
    assert any("fraud probability" in md.value for md in score.markdown)


def test_model_insights_renders_no_exception():
    at = AppTest.from_file(str(DASHBOARD_ROOT / "app_pages" / "model_insights.py"))
    at.run(timeout=30)
    assert not at.exception, [str(e) for e in at.exception]


def test_score_claim_persists_a_queryable_claim():
    """PB-12: a claim scored from the dashboard used to be shown and then
    discarded — nothing reached the DB, so it never showed up in
    `GET /claims`, the risk grid, or the audit log the way an API-scored
    claim did. Now it must be there, tagged ingested_via="dashboard"."""
    from app.db.models import AuditLogEntry, Claim, ScoredClaim
    from app.db.session import SessionLocal

    db = SessionLocal()
    try:
        before = db.query(Claim).filter(Claim.ingested_via == "dashboard").count()
    finally:
        db.close()

    at = AppTest.from_file(str(DASHBOARD_ROOT / "app_pages" / "score_claim.py"))
    at.run(timeout=30)
    at.button[0].click().run(timeout=30)
    assert not at.exception, [str(e) for e in at.exception]

    db = SessionLocal()
    try:
        dashboard_claims = db.query(Claim).filter(Claim.ingested_via == "dashboard").order_by(Claim.id.desc()).all()
        assert len(dashboard_claims) == before + 1
        newest = dashboard_claims[0]
        scored = db.query(ScoredClaim).filter(ScoredClaim.claim_id == newest.id).one_or_none()
        assert scored is not None
        audit = db.query(AuditLogEntry).filter(AuditLogEntry.claim_id == newest.id, AuditLogEntry.event_type == "claim_scored").one_or_none()
        assert audit is not None
    finally:
        db.close()

    saved_captions = [c.value for c in at.caption if "Saved as claim #" in c.value]
    assert saved_captions, "expected a 'Saved as claim #N' caption confirming persistence"


def test_batch_review_persists_claims_and_escalation():
    """PB-12: the scored batch and any escalation used to live only in
    st.session_state — gone on refresh, invisible to the API/audit log.
    Now both must land in the DB, and escalating must produce a
    claim_escalated audit-log entry against the claim's real id."""
    from app.db.models import AuditLogEntry, Claim, ScoredClaim
    from app.db.session import SessionLocal

    at = AppTest.from_file(str(DASHBOARD_ROOT / "app_pages" / "batch_review.py"))
    at.run(timeout=30)
    assert not at.exception, [str(e) for e in at.exception]

    raw = REPO_ROOT / "data" / "raw" / "insurance_claims_raw.csv"
    sample = pd.read_csv(raw).head(3)
    csv_bytes = sample.to_csv(index=False).encode()
    at.file_uploader[0].set_value(("batch_test.csv", csv_bytes, "text/csv")).run(timeout=30)
    assert not at.exception, [str(e) for e in at.exception]

    db = SessionLocal()
    try:
        batch_claims = db.query(Claim).filter(Claim.ingested_via == "dashboard_batch").all()
        assert len(batch_claims) >= 3
        batch_ids = [c.id for c in batch_claims]
        scored_count = db.query(ScoredClaim).filter(ScoredClaim.claim_id.in_(batch_ids)).count()
        assert scored_count == len(batch_claims), "every persisted claim should have a matching ScoredClaim row"
    finally:
        db.close()

    # Escalate the (pre-selected) first row.
    at.text_input[0].set_value("j.analyst").run(timeout=30)
    at.text_input[1].set_value("please take a look").run(timeout=30)
    at.button[0].click().run(timeout=30)
    assert not at.exception, [str(e) for e in at.exception]

    db = SessionLocal()
    try:
        entry = db.query(AuditLogEntry).filter(AuditLogEntry.event_type == "claim_escalated").order_by(AuditLogEntry.id.desc()).first()
        assert entry is not None
        assert entry.actor == "j.analyst"
        assert entry.detail == {"note": "please take a look"}
        assert entry.claim_id is not None
    finally:
        db.close()

    assert any("escalated for investigation" in s.value for s in at.success)


def test_overview_shows_bootstrap_ci_caption():
    """SH-04: the Overview page's KPI cards must carry a bootstrap 95% CI
    caption, not just a bare point estimate, when holdout_bootstrap_ci.csv
    exists (generated by `python -m app.ml.train`)."""
    bootstrap_path = REPO_ROOT / "data" / "processed" / "holdout_bootstrap_ci.csv"
    if not bootstrap_path.exists():
        import pytest
        pytest.skip("holdout_bootstrap_ci.csv not generated yet — run `python -m app.ml.train` first")
    at = AppTest.from_file(str(DASHBOARD_ROOT / "app_pages" / "overview.py"))
    at.run(timeout=30)
    assert not at.exception, [str(e) for e in at.exception]
    captions = [c.value for c in at.caption]
    kpi_captions = "\n".join(m.value for m in at.markdown)
    assert "95% CI" in kpi_captions or any("95% CI" in c for c in captions)
    assert any("5-fold CV" in c for c in captions)
