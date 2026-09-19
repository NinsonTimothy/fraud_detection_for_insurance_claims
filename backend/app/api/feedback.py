"""api/feedback.py — investigator feedback logger + retraining CSV export.

PB-07 (fixed): InvestigatorFeedback.claim_id is unique=True (one feedback
record per claim, by design), but a second submission for the same claim
used to raise an unhandled sqlalchemy.exc.IntegrityError -> 500.
Reproduced directly: submit feedback for a claim, submit again -> 500.
Fixed with a pre-insert existence check, same pattern as scoring.py's
duplicate-external_ref fix, returning a clean 409 instead.
"""
from __future__ import annotations

import io

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.schemas import FeedbackIn
from app.db.models import AuditLogEntry, Claim, InvestigatorFeedback
from app.db.session import get_db

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("")
def submit_feedback(feedback_in: FeedbackIn, db: Session = Depends(get_db)):
    claim = db.query(Claim).filter(Claim.id == feedback_in.claim_id).first()
    if not claim:
        raise HTTPException(404, "claim not found")

    existing = db.query(InvestigatorFeedback).filter(InvestigatorFeedback.claim_id == feedback_in.claim_id).first()
    if existing is not None:
        raise HTTPException(409, f"feedback was already submitted for claim_id={feedback_in.claim_id} (by {existing.investigator_name!r})")

    fb = InvestigatorFeedback(
        claim_id=feedback_in.claim_id, investigator_name=feedback_in.investigator_name,
        confirmed_fraud=feedback_in.confirmed_fraud, notes=feedback_in.notes,
    )
    db.add(fb)
    db.add(AuditLogEntry(
        event_type="feedback_submitted", claim_id=feedback_in.claim_id, actor=feedback_in.investigator_name,
        detail={"confirmed_fraud": feedback_in.confirmed_fraud},
    ))
    db.commit()
    return {"status": "recorded", "claim_id": feedback_in.claim_id}


@router.get("/export")
def export_feedback_csv(db: Session = Depends(get_db)):
    """Manual retraining loop (disclosed, not automatic — see
    docs/LIMITATIONS.md): export this CSV, append confirmed labels to the
    training set, re-run `python -m app.ml.train`."""
    rows = db.query(InvestigatorFeedback, Claim).join(Claim, Claim.id == InvestigatorFeedback.claim_id).all()
    records = [
        {**c.raw_payload, "fraud_reported": "Y" if fb.confirmed_fraud else "N",
         "investigator": fb.investigator_name, "submitted_at": fb.submitted_at.isoformat()}
        for fb, c in rows
    ]
    df = pd.DataFrame(records)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                              headers={"Content-Disposition": "attachment; filename=investigator_feedback_export.csv"})
