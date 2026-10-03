"""api/monitoring.py — PSI drift + KPI endpoints for the Monitoring dashboard page."""
from __future__ import annotations

import json

import pandas as pd
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.config import MODELS_DIR
from app.db.models import ScoredClaim
from app.db.session import get_db
from app.ml.psi import psi_report

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

PROCESSED_DIR = MODELS_DIR.parent / "data" / "processed"


@router.get("/kpis")
def kpis(db: Session = Depends(get_db)):
    with open(MODELS_DIR / "metrics.json") as f:
        metrics = json.load(f)
    total_scored = db.query(ScoredClaim).count()
    flagged = db.query(ScoredClaim).filter(ScoredClaim.flagged.is_(True)).count()
    return {
        "internal_holdout": metrics["model_comparison"][0],
        "operating_threshold": metrics["operating_threshold"],
        "total_scored_this_deployment": total_scored,
        "flagged_this_deployment": flagged,
    }


@router.get("/drift")
def drift(db: Session = Depends(get_db)):
    """PSI of live-scored probabilities vs. the internal test cohort's
    scored probabilities — a coarse, always-available drift signal even
    before enough live volume exists to PSI individual features."""
    reference = pd.read_csv(PROCESSED_DIR / "risk_scores_test.csv")
    live_rows = db.query(ScoredClaim.fraud_probability).all()
    if len(live_rows) < 30:
        return {"status": "insufficient_live_volume", "n_live": len(live_rows), "minimum_required": 30}
    live_df = pd.DataFrame({"y_proba": [r[0] for r in live_rows]})
    report = psi_report(reference, live_df, ["y_proba"])
    return report.to_dict(orient="records")
