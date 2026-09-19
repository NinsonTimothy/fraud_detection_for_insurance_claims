"""db/models.py — SQLAlchemy ORM models. Same schema whether the engine
behind it is Postgres (docker-compose) or SQLite (local/sandbox fallback)."""
from __future__ import annotations

import datetime as dt

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Claim(Base):
    __tablename__ = "claims"

    id = Column(Integer, primary_key=True, autoincrement=True)
    external_ref = Column(String(64), unique=True, index=True, nullable=True)
    raw_payload = Column(JSON, nullable=False)
    ingested_via = Column(String(32), nullable=False, default="api")  # api | batch_csv | kafka
    received_at = Column(DateTime, default=dt.datetime.utcnow, index=True)

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
    scored_at = Column(DateTime, default=dt.datetime.utcnow, index=True)

    claim = relationship("Claim", back_populates="score")


class InvestigatorFeedback(Base):
    __tablename__ = "investigator_feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    claim_id = Column(Integer, ForeignKey("claims.id"), unique=True, index=True)
    investigator_name = Column(String(128), nullable=False)
    confirmed_fraud = Column(Boolean, nullable=False)
    notes = Column(Text, nullable=True)
    submitted_at = Column(DateTime, default=dt.datetime.utcnow, index=True)

    claim = relationship("Claim", back_populates="feedback")


class AuditLogEntry(Base):
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String(64), nullable=False)  # claim_ingested | claim_scored | feedback_submitted | model_reloaded
    claim_id = Column(Integer, ForeignKey("claims.id"), nullable=True)
    detail = Column(JSON, nullable=True)
    actor = Column(String(64), nullable=False, default="system")
    created_at = Column(DateTime, default=dt.datetime.utcnow, index=True)
