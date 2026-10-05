"""
risk_policy.py — the ONE place a fraud-risk score becomes a display band, a
review flag and a recommended next step.

THREE DIFFERENT THRESHOLDS — never conflate them (B5):
  1. Display bands (Low / Medium / High): edges DERIVED from development
     out-of-fold scores by a documented rule (config.HIGH_BAND_FRAUD_CAPTURE /
     MEDIUM_BAND_FRAUD_CAPTURE) and written to models/risk_policy.json by
     train.py. They answer "how should this claim be prioritised?".
  2. Operational review threshold (`operating_threshold` in metrics.json):
     F1-optimal on development out-of-fold scores. Answers "flag for review,
     yes or no?".
  3. Cost-sensitivity thresholds (data/processed/cost_sensitivity.csv): a
     table of what the threshold WOULD be under different cost assumptions.
     Reported, never used to make decisions.

PB-04 (kept): scalar and vectorised grading use the same `>=` comparisons so
batch and single-claim scoring cannot disagree at a band edge.

DS-01 (kept, reworded per supervisor spec): the system recommends; a person
decides. No band leads to automatic approval or denial.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

POLICY_FILE = Path(__file__).resolve().parents[3] / "models" / "risk_policy.json"

# Fallback edges, used only if models/risk_policy.json does not exist yet
# (e.g. before the first training run). Real edges come from the artifact.
_FALLBACK = {"medium_edge": 0.3, "high_edge": 0.6}


def _load_edges() -> dict:
    try:
        with open(POLICY_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return dict(_FALLBACK)


_POLICY = _load_edges()
MEDIUM_RISK_EDGE: float = float(_POLICY["medium_edge"])
HIGH_RISK_EDGE: float = float(_POLICY["high_edge"])

RISK_GRADES = ("Low", "Medium", "High")

RECOMMENDED_ACTIONS = {
    "Low": "Low priority — standard processing; an investigator may still review",
    "Medium": "Medium priority — recommend investigator review",
    "High": "High priority — recommend early SIU review",
}

DECISION_SUPPORT_NOTICE = (
    "Decision support only: Aegis produces a fraud-risk score and a recommended priority. "
    "It does not approve, deny or settle any claim — an investigator decides."
)


def derive_band_edges(y_true, scores, high_capture: float, medium_capture: float) -> dict:
    """Edges such that, on the given (development, out-of-fold) scores, the
    High band is the smallest set of top-scored claims containing
    `high_capture` of all fraud, and High+Medium contain `medium_capture`.
    Ties at an edge all fall into the higher band (>= comparison)."""
    y, s = np.asarray(y_true), np.asarray(scores, dtype=float)
    order = np.argsort(-s, kind="stable")
    cum = np.cumsum(y[order]) / max(1, y.sum())

    def edge(capture):
        k = int(np.searchsorted(cum, capture - 1e-12))
        return float(s[order][min(k, len(s) - 1)])

    high, medium = edge(high_capture), edge(medium_capture)
    medium = min(medium, high)
    report = {}
    for name, lo, hi in (("High", high, np.inf), ("Medium", medium, high), ("Low", -np.inf, medium)):
        m = (s >= lo) & (s < hi)
        report[name] = {"share_of_claims": float(m.mean()), "share_of_fraud": float(y[m].sum() / max(1, y.sum())),
                        "fraud_rate_in_band": float(y[m].mean()) if m.any() else None}
    return {"medium_edge": medium, "high_edge": high, "rule": {
        "high_band_fraud_capture": high_capture, "medium_band_fraud_capture": medium_capture,
        "description": (f"High = top-scored development claims containing {high_capture:.0%} of fraud; "
                        f"High + Medium contain {medium_capture:.0%}; derived from out-of-fold scores on the "
                        "development split only.")}, "development_band_profile": report}


def grade_for(proba: float, edges: dict | None = None) -> str:
    med = edges["medium_edge"] if edges else MEDIUM_RISK_EDGE
    high = edges["high_edge"] if edges else HIGH_RISK_EDGE
    if proba >= high:
        return "High"
    if proba >= med:
        return "Medium"
    return "Low"


def grade_for_array(proba, edges: dict | None = None) -> np.ndarray:
    med = edges["medium_edge"] if edges else MEDIUM_RISK_EDGE
    high = edges["high_edge"] if edges else HIGH_RISK_EDGE
    proba = np.asarray(proba)
    return np.select([proba >= high, proba >= med], ["High", "Medium"], default="Low")


def is_flagged(proba: float, operating_threshold: float) -> bool:
    return proba >= operating_threshold


def recommended_action(grade: str) -> str:
    return RECOMMENDED_ACTIONS.get(grade, RECOMMENDED_ACTIONS["Medium"])


def band_table(edges: dict | None = None) -> pd.DataFrame:
    med = edges["medium_edge"] if edges else MEDIUM_RISK_EDGE
    high = edges["high_edge"] if edges else HIGH_RISK_EDGE
    return pd.DataFrame([{"band": "Low", "from": 0.0, "to": med}, {"band": "Medium", "from": med, "to": high},
                         {"band": "High", "from": high, "to": 1.0}])
