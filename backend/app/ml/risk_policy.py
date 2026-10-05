"""
risk_policy.py — PB-04: the ONE place a fraud probability turns into a
risk band, a flag/no-flag decision, and a recommended next action. Before
this module existed, `inference.py`'s `score_batch()` (vectorized, via
`pd.cut`) and `score_one()` (scalar, via hand-written if/elif) each
re-implemented the Low/Medium/High banding independently, and they
DISAGREED at the exact band edges:

    proba=0.30 -> score_batch: "Low"    score_one: "Medium"   (mismatch)
    proba=0.60 -> score_batch: "Medium" score_one: "High"     (mismatch)

`pd.cut(bins=[-0.01, 0.3, 0.6, 1.01])` is right-inclusive by default
((a, b] intervals), so 0.30 lands in the FIRST bin ("Low"); the
hand-written `"High" if p >= 0.6 else "Medium" if p >= 0.3 else "Low"`
puts 0.30 in "Medium" because `>=` includes the boundary on the LOWER
side instead. Same probability, same claim, two different answers
depending on whether it went through batch or single-claim scoring — a
real, reproducible bug (see `backend/tests/test_risk_policy.py`).

Fix: a single `grade_for()` used by both call sites (batch calls it via
`np.vectorize`-free array math in `grade_for_array()`, which uses the
exact same `>=` comparisons as the scalar version, so there is no second
implementation to drift out of sync).

The risk-band edges (0.3 / 0.6) are DELIBERATELY separate from
`operating_threshold` (the flag/no-flag cutoff, chosen by
`train.py`'s F1-optimal search — see PB-03): `flagged` answers "does an
analyst need to look at this at all", while `risk_grade` is a coarser,
fixed-edge display banding for the dashboard/API to sort claims by. They
are not supposed to be the same number, but both belong in this one
module so nothing else has to know the edges to compute either of them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Risk-band edges: a claim scoring >= HIGH_RISK_EDGE is "High" risk,
# >= MEDIUM_RISK_EDGE (and below HIGH_RISK_EDGE) is "Medium", else "Low".
# Fixed display bands, independent of the tuned operating_threshold.
MEDIUM_RISK_EDGE = 0.3
HIGH_RISK_EDGE = 0.6

RISK_GRADES = ("Low", "Medium", "High")

# DS-01 (pre-defence fix): decision-SUPPORT wording. The Low band used to
# read "No action — auto-approved", which described the system as making a
# claims decision on its own. It never does and must not: the model only
# RECOMMENDS a level of scrutiny, and a human (claims handler or
# investigator) makes every decision. Every action below is therefore
# phrased as a recommendation, and DECISION_SUPPORT_NOTICE is shown next to
# every score in the API response and the dashboard.
RECOMMENDED_ACTIONS = {
    "Low": "Recommend standard claims handling (no investigation suggested) — handler decides",
    "Medium": "Recommend investigator review — investigator decides",
    "High": "Recommend priority SIU review — investigator decides",
}

DECISION_SUPPORT_NOTICE = (
    "Decision support only: Aegis recommends a level of scrutiny. It does not approve, "
    "deny, or settle any claim; a human investigator makes every decision."
)


def grade_for(proba: float) -> str:
    """Scalar risk grade for a single fraud probability."""
    if proba >= HIGH_RISK_EDGE:
        return "High"
    if proba >= MEDIUM_RISK_EDGE:
        return "Medium"
    return "Low"


def grade_for_array(proba: np.ndarray | pd.Series) -> np.ndarray:
    """Vectorized risk grade for an array of fraud probabilities. Uses the
    exact same `>=` comparisons as `grade_for()` (not `pd.cut`, whose
    default right-inclusive bins disagree with `grade_for()` at the band
    edges — see module docstring) so batch and single-claim scoring can
    never diverge at a boundary value again."""
    proba = np.asarray(proba)
    return np.select(
        [proba >= HIGH_RISK_EDGE, proba >= MEDIUM_RISK_EDGE],
        ["High", "Medium"],
        default="Low",
    )


def is_flagged(proba: float, operating_threshold: float) -> bool:
    """Whether a claim needs analyst attention at all — deliberately a
    separate decision from `grade_for()`'s display banding (see module
    docstring)."""
    return proba >= operating_threshold


def recommended_action(grade: str) -> str:
    """Plain-language next step for a risk grade, for the dashboard/API to
    show alongside the probability and band — not persisted to the
    database (computed fresh from `risk_grade` wherever it's displayed)."""
    return RECOMMENDED_ACTIONS.get(grade, RECOMMENDED_ACTIONS["Medium"])
