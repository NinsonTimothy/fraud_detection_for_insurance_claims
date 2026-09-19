"""test_kafka_logic.py — verifies the Kafka producer/consumer LOGIC
directly in Python against the in-memory stub, since no live broker is
reachable in this build sandbox (see docs/LIMITATIONS.md)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.kafka.producer_sim import InMemoryKafkaStub, consume_and_process, produce_claim, validate_claim_message


def test_validate_claim_message_accepts_minimum_viable_claim():
    ok, err = validate_claim_message({"total_claim_amount": 5000, "incident_severity": "Minor Damage"})
    assert ok
    assert err is None


def test_validate_claim_message_rejects_missing_required_fields():
    ok, err = validate_claim_message({"total_claim_amount": 5000})
    assert not ok
    assert "incident_severity" in err


def test_validate_claim_message_rejects_unknown_fields_as_likely_typo():
    ok, err = validate_claim_message({"total_claim_amount": 5000, "incident_severity": "Minor Damage", "totall_claim": 1})
    assert not ok
    assert "totall_claim" in err


def test_produce_rejects_invalid_message():
    stub = InMemoryKafkaStub()
    with pytest.raises(ValueError):
        produce_claim(stub, "t", {"bad": "payload"})
    assert len(stub._queue) == 0


def test_end_to_end_produce_then_consume():
    stub = InMemoryKafkaStub()
    produce_claim(stub, "t", {"total_claim_amount": 12000, "incident_severity": "Major Damage"})
    produce_claim(stub, "t", {"total_claim_amount": 3000, "incident_severity": "Minor Damage"})

    processed = []
    n = consume_and_process(stub, lambda msg: processed.append(msg))
    assert n == 2
    assert len(processed) == 2
    assert stub._queue == []  # drained
