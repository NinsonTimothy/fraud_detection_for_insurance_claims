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

PB-26 (fixed): found by live-testing `/score/batch` end-to-end (not by
pytest — see this ticket's note in docs/REBUILD_NOTES.md for exactly how
it slipped past the existing suite). A batch CSV row with ANY missing
optional field (e.g. `total_claim_amount` simply blank — a normal, every-
day case PB-20 explicitly made legal, not an edge case) round-trips
through `pd.read_csv(..., na_values=[""])` as a real float NaN. Nothing
before this fix ever converted that NaN to `None` before it was written
into `Claim.raw_payload` (a JSON column) — `json.dumps`'s default
`allow_nan=True` lets it write successfully, so the INSERT itself never
errored. The break was on READ: `GET /claims` and `GET /claims/{id}`
re-serialize that same payload through Starlette's default
`JSONResponse`, which uses `allow_nan=False` (RFC 8259-compliant JSON has
no NaN/Infinity token at all) — so every claim with so much as one
blank optional field in its original batch upload 500'd the instant
anyone tried to look at it again, even though scoring itself had
succeeded and returned 200. Reproduced directly: POST a 2-row batch CSV
where row 2 leaves `total_claim_amount`/dates blank -> 200 -> GET
`/claims/{that claim's id}` -> 500 (`ValueError: Out of range float
values are not JSON compliant`). Fixed by sanitizing NaN/NaT to `None`
in `_sanitize_for_json()` below, applied to every payload at the ONE
place both `/score/batch` and the dashboard's Batch review page write
through — the same "one shared choke point, not two copies that can
drift" reasoning PB-12 already established for this module. `/score`
(single-claim) was never affected — `ClaimPayload.model_dump(exclude_none=True)`
already leaves a missing field genuinely ABSENT from the dict rather than
present-with-NaN, so this sanitization is a no-op for that path, applied
here anyway for defense in depth rather than relying on two different
code paths independently avoiding the same mistake.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.models import AuditLogEntry, Claim, ScoredClaim


def _sanitize_for_json(value):
    """Recursively replace a NaN/Infinity float or a pandas NaT with
    `None`, so nothing that entered via a pandas batch-scoring path (see
    this module's PB-26 docstring section) is ever persisted in a form
    that Starlette's default (RFC-compliant, `allow_nan=False`)
    `JSONResponse` refuses to re-serialize later. Dependency-free: NaN and
    NaT are the only values in Python for which `x != x` is true, so this
    needs no pandas/numpy import to detect either one, and works
    unchanged if a future raw field is ever a pandas Timestamp/NaT rather
    than a plain float."""
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(v) for v in value]
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    try:
        if value != value:  # catches pandas NaT (and any other NaN-like sentinel) without importing pandas
            return None
    except Exception:
        pass
    return value


def persist_scored_claim(
    db: Session, payload: dict, result: dict, *, external_ref: str | None = None, ingested_via: str = "api",
) -> Claim:
    """One Claim + one ScoredClaim + one audit entry. Caller commits (kept
    separate so a caller doing several writes in one request/session can
    batch them into a single commit, same as scoring.py already does)."""
    claim = Claim(external_ref=external_ref, raw_payload=_sanitize_for_json(payload), ingested_via=ingested_via)
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
    claims = [Claim(raw_payload=_sanitize_for_json(row), ingested_via=ingested_via) for row in raw_rows]
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
