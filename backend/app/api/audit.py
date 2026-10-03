"""api/audit.py — compliance timeline read endpoint."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.db.models import AuditLogEntry
from app.db.session import get_db

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("")
def audit_timeline(limit: int = 200, db: Session = Depends(get_db)):
    rows = db.query(AuditLogEntry).order_by(desc(AuditLogEntry.created_at)).limit(limit).all()
    return [
        {"id": r.id, "event_type": r.event_type, "claim_id": r.claim_id, "actor": r.actor,
         "detail": r.detail, "created_at": r.created_at.isoformat()}
        for r in rows
    ]
