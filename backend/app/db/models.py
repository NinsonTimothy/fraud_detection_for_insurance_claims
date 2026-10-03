"""db/models.py — SQLAlchemy ORM models. Same schema whether the engine
behind it is Postgres (docker-compose) or SQLite (local/sandbox fallback)."""
from __future__ import annotations

import datetime as dt

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def utcnow() -> dt.datetime:
    """Timezone-aware replacement for the deprecated `datetime.utcnow()`
    (PB-22). Used as every timestamp column's `default=` callable below."""
    return dt.datetime.now(dt.timezone.utc)


class Claim(Base):
    __tablename__ = "claims"

    id = Column(Integer, primary_key=True, autoincrement=True)
    external_ref = Column(String(64), unique=True, index=True, nullable=True)
    raw_payload = Column(JSON, nullable=False)
    # PB-08/PB-09 (D1): "kafka" removed from the valid-values comment — the
    # Kafka ingestion path is gone entirely, not just unused. The column
    # itself (and any historical "kafka" rows a pre-existing DB might
    # still have) is left alone; this is a comment/behavior change, not a
    # migration.
    # PB-12: "dashboard" / "dashboard_batch" added — the dashboard's
    # Score/Batch review pages now persist through the same DB every other
    # ingestion path uses (db/persistence.py), tagged distinctly from
    # "api"/"batch_csv" so the audit trail can tell which surface a claim
    # actually came through.
    ingested_via = Column(String(32), nullable=False, default="api")  # api | batch_csv | dashboard | dashboard_batch
    received_at = Column(DateTime(timezone=True), default=utcnow, index=True)

    score = relationship("ScoredClaim", back_populates="claim", uselist=False)
    feedback = relationship("InvestigatorFeedback", back_populates="claim", uselist=False)


class ScoredClaim(Base):
    __tablename__ = "scored_claims"

    id = Column(Integer, primary_key=True, autoincrement=True)
    claim_id = Column(Integer, ForeignKey("claims.id"), unique=True, index=True)
    fraud_probability = Column(Float, nullable=False)
    risk_grade = Column(String(16), nullable=False)  # Low | Medium | High
    flagged = Column(Boolean, nullable=False)
    operating_threshold = Column(Float, nullable=False)
    top_reasons = Column(JSON, nullable=True)
    model_version = Column(String(64), nullable=False)
    scored_at = Column(DateTime(timezone=True), default=utcnow, index=True)

    claim = relationship("Claim", back_populates="score")


class InvestigatorFeedback(Base):
    __tablename__ = "investigator_feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    claim_id = Column(Integer, ForeignKey("claims.id"), unique=True, index=True)
    investigator_name = Column(String(128), nullable=False)
    confirmed_fraud = Column(Boolean, nullable=False)
    notes = Column(Text, nullable=True)
    submitted_at = Column(DateTime(timezone=True), default=utcnow, index=True)

    claim = relationship("Claim", back_populates="feedback")


class AuditLogEntry(Base):
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String(64), nullable=False)  # claim_ingested | claim_scored | feedback_submitted | model_reloaded
    claim_id = Column(Integer, ForeignKey("claims.id"), nullable=True)
    detail = Column(JSON, nullable=True)
    actor = Column(String(64), nullable=False, default="system")
    created_at = Column(DateTime(timezone=True), default=utcnow, index=True)
