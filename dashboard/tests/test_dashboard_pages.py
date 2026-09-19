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
