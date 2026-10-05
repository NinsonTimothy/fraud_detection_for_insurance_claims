"""
sensitivity_probe.py — B9: score a REFERENCE claim while varying one field at
a time, holding everything else fixed. Shows how the shipped champion
actually responds to each input (e.g. severity: Trivial / Minor / Major /
Total Loss) so non-monotone or surprising behaviour is visible and disclosed.

Reference claim = feature_engineering.MISSING_COLUMN_DEFAULTS (the documented
typical values). Output: data/processed/sensitivity_probe.csv
Run: python -m app.ml.sensitivity_probe
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.ml.feature_engineering import MISSING_COLUMN_DEFAULTS
from app.ml.inference import FraudScoringService

OUT = Path(__file__).resolve().parents[3] / "data" / "processed" / "sensitivity_probe.csv"


def _variants() -> list[tuple[str, str, dict]]:
    base = dict(MISSING_COLUMN_DEFAULTS)
    v = []
    for sev in ("Trivial Damage", "Minor Damage", "Major Damage", "Total Loss"):
        v.append(("incident_severity", sev, {"incident_severity": sev}))
    for w in (0, 1, 2, 3):
        v.append(("witnesses", str(w), {"witnesses": w}))
    for pr in ("NO", "YES"):
        v.append(("police_report_available", pr, {"police_report_available": pr}))
    for f in (0.5, 1.0, 1.5, 2.0):
        parts = {k: base[k] * f for k in ("injury_claim", "property_claim", "vehicle_claim")}
        v.append(("claim_amounts", f"x{f}", {**parts, "total_claim_amount": sum(parts.values())}))
    for years in (0, 1, 5, 10, 20):
        bind = (pd.Timestamp(base["incident_date"]) - pd.DateOffset(years=years) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        v.append(("policy_bind_date", f"{years}y before incident", {"policy_bind_date": bind}))
    for vy in (2000, 2005, 2010, 2015):
        v.append(("auto_year", str(vy), {"auto_year": vy}))
    return v


def run() -> pd.DataFrame:
    svc = FraudScoringService.instance()
    rows = []
    for field, label, change in _variants():
        claim = {**MISSING_COLUMN_DEFAULTS, **change}
        r = svc.score_one(claim)
        rows.append({"field": field, "value": label, "fraud_risk_score": r["fraud_probability"],
                     "risk_band": r["risk_grade"], "flagged_for_review": r["flagged"],
                     "top_factor": r["top_reasons"][0]["sentence"] if r["top_reasons"] else ""})
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    return df


if __name__ == "__main__":
    print(run().to_string(index=False))
