"""
api/ingestion.py — three ways a claim can enter the system: a single JSON
POST (see scoring.py's /score, which ingests+scores together), a batch CSV
upload (scoring.py's /score/batch), and a simulated Kafka stream — the
producer/consumer LOGIC is real and tested (app/kafka/producer_sim.py),
wired to an in-memory stub in this sandbox build (no live broker reachable
— see docs/LIMITATIONS.md); pointed at a real broker in docker-compose,
the same call shape drives `confluent_kafka.Producer`/`Consumer` instead.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.models import AuditLogEntry, Claim, ScoredClaim
from app.db.session import get_db
from app.kafka.producer_sim import InMemoryKafkaStub, consume_and_process, produce_claim
from app.ml.inference import FraudScoringService

router = APIRouter(prefix="/ingest", tags=["ingestion"])

# Module-level stub standing in for a live Kafka topic in this build.
_claims_topic = InMemoryKafkaStub()


@router.post("/kafka/produce")
def produce_to_stream(payload: dict):
    try:
        produce_claim(_claims_topic, "aegis.claims.raw", payload)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"status": "queued", "queue_depth": len(_claims_topic._queue)}


@router.post("/kafka/consume")
def consume_stream(db: Session = Depends(get_db)):
    service = FraudScoringService.instance()
    scored_ids = []

    def handler(payload: dict):
        claim = Claim(raw_payload=payload, ingested_via="kafka")
        db.add(claim)
        db.flush()
        result = service.score_one(payload)
        db.add(ScoredClaim(
            claim_id=claim.id, fraud_probability=result["fraud_probability"], risk_grade=result["risk_grade"],
            flagged=result["flagged"], operating_threshold=result["operating_threshold"],
            top_reasons=result["top_reasons"], model_version=result["model_version"],
        ))
        db.add(AuditLogEntry(event_type="claim_ingested", claim_id=claim.id, detail={"via": "kafka"}))
        scored_ids.append(claim.id)

    n_processed = consume_and_process(_claims_topic, handler)
    db.commit()
    return {"n_processed": n_processed, "claim_ids": scored_ids}
