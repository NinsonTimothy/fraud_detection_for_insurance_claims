"""test_persistence.py — db/persistence.py exercised directly against a
throwaway SQLite DB (no FastAPI app involved).

PB-12: this module is the ONLY place that writes a Claim/ScoredClaim/
AuditLogEntry now — api/scoring.py's /score and /score/batch, and the
dashboard's Score a claim / Batch review pages, all call into it instead
of each hand-rolling the same three-table write. These tests are the
single source of truth for that write logic; test_api.py's existing
/score and /score/batch tests keep covering the HTTP-layer behavior
(status codes, response shape) built on top of it.
"""
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.models import AuditLogEntry, Base, Claim, ScoredClaim
from app.db.persistence import persist_escalation, persist_scored_claim, persist_scored_claims_batch

SAMPLE_PAYLOAD = {"age": 35, "insured_zip": 468000}
SAMPLE_RESULT = {
    "fraud_probability": 0.42, "risk_grade": "Medium", "flagged": False,
    "operating_threshold": 0.5, "top_reasons": [{"feature": "age", "sentence": "..."}],
    "model_version": "random_forest-v1",
}


@pytest.fixture()
def db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/persistence_test.db", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(autocommit=False, autoflush=False, bind=engine)()
    try:
        yield session
    finally:
        session.close()


class TestPersistScoredClaim:
    def test_creates_claim_scoredclaim_and_audit_entry(self, db):
        claim = persist_scored_claim(db, SAMPLE_PAYLOAD, SAMPLE_RESULT, external_ref="ext-1", ingested_via="api")
        db.commit()

        assert claim.id is not None
        assert db.query(Claim).count() == 1
        stored_claim = db.query(Claim).one()
        assert stored_claim.external_ref == "ext-1"
        assert stored_claim.ingested_via == "api"
        assert stored_claim.raw_payload == SAMPLE_PAYLOAD

        scored = db.query(ScoredClaim).filter(ScoredClaim.claim_id == claim.id).one()
        assert scored.fraud_probability == pytest.approx(0.42)
        assert scored.risk_grade == "Medium"
        assert scored.flagged is False
        assert scored.model_version == "random_forest-v1"
        assert scored.top_reasons == SAMPLE_RESULT["top_reasons"]

        audit = db.query(AuditLogEntry).filter(AuditLogEntry.claim_id == claim.id).one()
        assert audit.event_type == "claim_scored"
        assert audit.detail["ingested_via"] == "api"

    def test_defaults_ingested_via_to_api_and_external_ref_to_none(self, db):
        claim = persist_scored_claim(db, SAMPLE_PAYLOAD, SAMPLE_RESULT)
        db.commit()
        assert claim.external_ref is None
        assert claim.ingested_via == "api"

    def test_dashboard_ingested_via_is_tagged_distinctly(self, db):
        claim = persist_scored_claim(db, SAMPLE_PAYLOAD, SAMPLE_RESULT, ingested_via="dashboard")
        db.commit()
        assert db.get(Claim, claim.id).ingested_via == "dashboard"


class TestPersistScoredClaimsBatch:
    def test_creates_one_row_per_claim(self, db):
        raw_rows = [{"age": 20}, {"age": 30}, {"age": 40}]
        scored_rows = [{**SAMPLE_RESULT, "fraud_probability": p} for p in (0.1, 0.5, 0.9)]

        claim_ids = persist_scored_claims_batch(db, raw_rows, scored_rows, ingested_via="dashboard_batch")
        db.commit()

        assert len(claim_ids) == 3
        assert len(set(claim_ids)) == 3  # distinct ids, not the same row reused
        assert db.query(Claim).count() == 3
        assert db.query(ScoredClaim).count() == 3
        for cid in claim_ids:
            assert db.get(Claim, cid).ingested_via == "dashboard_batch"

        probs = {s.claim_id: s.fraud_probability for s in db.query(ScoredClaim).all()}
        assert sorted(probs.values()) == pytest.approx([0.1, 0.5, 0.9])

        audit = db.query(AuditLogEntry).filter(AuditLogEntry.event_type == "batch_scored").one()
        assert audit.detail["n_rows"] == 3
        assert audit.detail["ingested_via"] == "dashboard_batch"

    def test_raises_on_length_mismatch(self, db):
        with pytest.raises(ValueError, match="same length"):
            persist_scored_claims_batch(db, [{"age": 20}, {"age": 30}], [SAMPLE_RESULT])

    def test_empty_batch_is_a_noop_not_an_error(self, db):
        claim_ids = persist_scored_claims_batch(db, [], [])
        db.commit()
        assert claim_ids == []
        assert db.query(Claim).count() == 0


class TestPersistEscalation:
    def test_creates_claim_escalated_audit_entry(self, db):
        claim = persist_scored_claim(db, SAMPLE_PAYLOAD, SAMPLE_RESULT)
        db.flush()

        entry = persist_escalation(db, claim.id, "j.analyst", "looks staged")
        db.commit()

        assert entry.event_type == "claim_escalated"
        assert entry.claim_id == claim.id
        assert entry.actor == "j.analyst"
        assert entry.detail == {"note": "looks staged"}

    def test_does_not_create_investigator_feedback(self, db):
        """Escalation is a lighter-weight "look at this" signal, not a
        confirmed_fraud determination — it must never be mistaken for real
        investigator feedback (api/feedback.py), which feeds retraining."""
        claim = persist_scored_claim(db, SAMPLE_PAYLOAD, SAMPLE_RESULT)
        db.flush()
        persist_escalation(db, claim.id, "j.analyst", "note")
        db.commit()

        from app.db.models import InvestigatorFeedback
        assert db.query(InvestigatorFeedback).count() == 0

    def test_blank_investigator_name_falls_back_to_dashboard_analyst(self, db):
        claim = persist_scored_claim(db, SAMPLE_PAYLOAD, SAMPLE_RESULT)
        db.flush()
        entry = persist_escalation(db, claim.id, "", "note")
        db.commit()
        assert entry.actor == "dashboard_analyst"
