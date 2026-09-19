"""api/claims.py — claims list / risk-grid view for the review queue."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.db.models import Claim, ScoredClaim
from app.db.session import get_db

router = APIRouter(prefix="/claims", tags=["claims"])


@router.get("")
def list_claims(risk_grade: str | None = None, flagged_only: bool = False, limit: int = 200, db: Session = Depends(get_db)):
    q = db.query(Claim, ScoredClaim).join(ScoredClaim, ScoredClaim.claim_id == Claim.id)
    if risk_grade:
        q = q.filter(ScoredClaim.risk_grade == risk_grade)
    if flagged_only:
        q = q.filter(ScoredClaim.flagged.is_(True))
    rows = q.order_by(desc(ScoredClaim.fraud_probability)).limit(limit).all()
    return [
        {
            "claim_id": c.id, "external_ref": c.external_ref, "received_at": c.received_at.isoformat(),
            "fraud_probability": s.fraud_probability, "risk_grade": s.risk_grade, "flagged": s.flagged,
            "raw_payload": c.raw_payload,
        }
        for c, s in rows
    ]


@router.get("/{claim_id}")
def get_claim(claim_id: int, db: Session = Depends(get_db)):
    claim = db.query(Claim).filter(Claim.id == claim_id).first()
    if not claim:
        raise HTTPException(404, "claim not found")
    score = claim.score
    return {
        "claim_id": claim.id, "raw_payload": claim.raw_payload, "ingested_via": claim.ingested_via,
        "received_at": claim.received_at.isoformat(),
        "score": None if not score else {
            "fraud_probability": score.fraud_probability, "risk_grade": score.risk_grade,
            "flagged": score.flagged, "top_reasons": score.top_reasons, "model_version": score.model_version,
        },
    }


@router.get("/risk-grid/summary")
def risk_grid_summary(db: Session = Depends(get_db)):
    rows = db.query(ScoredClaim).all()
    grid = {"Low": 0, "Medium": 0, "High": 0}
    for r in rows:
        grid[r.risk_grade] = grid.get(r.risk_grade, 0) + 1
    return {"total_scored": len(rows), "by_risk_grade": grid, "flagged": sum(1 for r in rows if r.flagged)}
