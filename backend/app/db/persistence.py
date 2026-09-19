"""
db/persistence.py — shared Claim/ScoredClaim/AuditLogEntry write logic.

PB-12 (fixed): the dashboard's "Score a claim" and "Batch review" pages
called `FraudScoringService.score_one()`/`score_batch()` directly and
threw the result away once the Streamlit session ended — nothing was
ever written to the DB. A claim scored through the API showed up in
`GET /claims`, the risk grid, and the audit log; the exact same claim
scored through the dashboard vanished. "Escalate a claim for
investigation" on the batch review page was worse: it only ever wrote to
`st.session_state`, so escalations didn't survive a page refresh, let
alone show up anywhere another analyst (or the API) could see them.

This module centralizes the DB-write logic `api/scoring.py`'s `/score`
already had, so the dashboard can call the SAME functions instead of
duplicating (and risking drifting from) that logic a second time — the
same reasoning that already governs `inference.py`'s shared
`FraudScoringService` (the API and dashboard can't disagree about a
SCORE; this makes sure they also can't disagree about how a score gets
RECORDED).
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.models import AuditLogEntry, Claim, ScoredClaim


def persist_scored_claim(
    db: Session, payload: dict, result: dict, *, external_ref: str | None = None, ingested_via: str = "api",
) -> Claim:
    """One Claim + one ScoredClaim + one audit entry. Caller commits (kept
    separate so a caller doing several writes in one request/session can
    batch them into a single commit, same as scoring.py already does)."""
    claim = Claim(external_ref=external_ref, raw_payload=payload, ingested_via=ingested_via)
    db.add(claim)
    db.flush()  # assigns claim.id
    db.add(ScoredClaim(
        claim_id=claim.id, fraud_probability=result["fraud_probability"], risk_grade=result["risk_grade"],
        flagged=result["flagged"], operating_threshold=result["operating_threshold"],
        top_reasons=result.get("top_reasons"), model_version=result["model_version"],
    ))
    db.add(AuditLogEntry(
        event_type="claim_scored", claim_id=claim.id,
        detail={"fraud_probability": result["fraud_probability"], "ingested_via": ingested_via},
    ))
    return claim


def persist_scored_claims_batch(
    db: Session, raw_rows: list[dict], scored_rows: list[dict], *, ingested_via: str = "dashboard_batch",
) -> list[int]:
    """Batch counterpart for the dashboard's Batch review page.
    `raw_rows`/`scored_rows` are parallel lists (same order, same length —
    one dict per claim). Written with ONE flush for all Claim rows (to
    assign every `.id` in a single round of work) and ONE `add_all` for
    every ScoredClaim, rather than a flush per row — the dashboard's
    batch page is new code as of this ticket, so it starts from the
    efficient pattern directly rather than the API's `/score/batch`
    per-row-flush loop (a separate, already-flagged issue there — PB-18 —
    that this ticket does not touch)."""
    if len(raw_rows) != len(scored_rows):
        raise ValueError(f"raw_rows ({len(raw_rows)}) and scored_rows ({len(scored_rows)}) must be the same length")
    claims = [Claim(raw_payload=row, ingested_via=ingested_via) for row in raw_rows]
    db.add_all(claims)
    db.flush()
    scored_objs = [
        ScoredClaim(
            claim_id=claim.id, fraud_probability=s["fraud_probability"], risk_grade=s["risk_grade"],
            flagged=s["flagged"], operating_threshold=s["operating_threshold"],
            top_reasons=s.get("top_reasons"), model_version=s["model_version"],
        )
        for claim, s in zip(claims, scored_rows)
    ]
    db.add_all(scored_objs)
    db.add(AuditLogEntry(event_type="batch_scored", detail={"n_rows": len(claims), "ingested_via": ingested_via}))
    return [c.id for c in claims]


def persist_escalation(db: Session, claim_id: int, investigator_name: str, note: str) -> AuditLogEntry:
    """"Escalate for investigation" is deliberately NOT `InvestigatorFeedback`
    (a ground-truth confirmed_fraud determination that feeds the
    retraining export — see api/feedback.py) — it's a lighter-weight "a
    human should look at this", so it's recorded as its own audit-log
    event type instead of presuming a fraud/not-fraud verdict the way
    real feedback does."""
    entry = AuditLogEntry(
        event_type="claim_escalated", claim_id=claim_id, actor=investigator_name or "dashboard_analyst",
        detail={"note": note},
    )
    db.add(entry)
    return entry
