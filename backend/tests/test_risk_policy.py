"""test_risk_policy.py — regression test for PB-04: score_batch() and
score_one() used to compute the risk_grade band independently (pd.cut's
right-inclusive bins vs. a hand-written >= if/elif) and DISAGREED at the
exact band edges (0.3 and 0.6). risk_policy.py is now the single source
of truth both call sites use."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.risk_policy import (
    HIGH_RISK_EDGE,
    MEDIUM_RISK_EDGE,
    grade_for,
    grade_for_array,
    is_flagged,
    recommended_action,
)


@pytest.mark.parametrize("proba", [0.0, 0.1, MEDIUM_RISK_EDGE - 1e-9, MEDIUM_RISK_EDGE, MEDIUM_RISK_EDGE + 1e-9, 0.45, HIGH_RISK_EDGE - 1e-9, HIGH_RISK_EDGE, HIGH_RISK_EDGE + 1e-9, 0.99, 1.0])
def test_scalar_and_vectorized_grade_agree_at_every_boundary(proba):
    """The exact bug this ticket reproduces: at proba==0.3 the old
    pd.cut()-based score_batch() said "Low" while the old hand-written
    score_one() said "Medium" (same for 0.6 -> Medium vs. High)."""
    scalar = grade_for(proba)
    vectorized = grade_for_array(np.array([proba]))[0]
    assert scalar == vectorized, f"grade_for({proba})={scalar!r} but grade_for_array disagreed: {vectorized!r}"


def test_grade_edges_match_the_documented_boundaries():
    assert grade_for(MEDIUM_RISK_EDGE) == "Medium"
    assert grade_for(np.nextafter(MEDIUM_RISK_EDGE, 0)) == "Low"
    assert grade_for(HIGH_RISK_EDGE) == "High"
    assert grade_for(np.nextafter(HIGH_RISK_EDGE, 0)) == "Medium"


def test_is_flagged_uses_operating_threshold_not_band_edges():
    # flagged is a SEPARATE decision from risk_grade — an operating
    # threshold below the Medium edge should flag some "Low"-banded claims.
    assert is_flagged(0.1, operating_threshold=0.05) is True
    assert grade_for(0.1) == "Low"
    assert is_flagged(0.1, operating_threshold=0.2) is False


def test_recommended_action_covers_every_grade():
    for grade in ("Low", "Medium", "High"):
        action = recommended_action(grade)
        assert isinstance(action, str) and action
