"""
producer_sim.py / consumer.py — Kafka producer+consumer logic for the
"claims ingested as a stream" transport (`docker-compose.yml`'s `kafka`
service, KRaft mode, no Zookeeper needed).

Disclosed limitation (see docs/LIMITATIONS.md): no live Kafka broker was
reachable in the build sandbox this was assembled in (Docker Hub is
network-blocked there, so no image — Kafka included — could be pulled).
This logic is verified directly in Python against an in-memory stand-in
queue (see backend/tests/test_kafka_logic.py) — the schema validation,
DB-write pipeline, and audit-log entry are real and tested; only the actual
`confluent-kafka`/`aiokafka` client wiring against a live broker is
unverified. `docker compose config` validates the service definition's
syntax.
"""
from __future__ import annotations

import json
from typing import Callable

from app.ml.feature_engineering import RAW_FEATURE_COLUMNS

REQUIRED_KEYS = {"total_claim_amount", "incident_severity"}  # minimum viable claim


def validate_claim_message(payload: dict) -> tuple[bool, str | None]:
    if not isinstance(payload, dict):
        return False, "payload is not a JSON object"
    missing_required = REQUIRED_KEYS - payload.keys()
    if missing_required:
        return False, f"missing required fields: {sorted(missing_required)}"
    unknown = set(payload.keys()) - set(RAW_FEATURE_COLUMNS) - {"external_ref"}
    if unknown:
        return False, f"unrecognized fields (typo?): {sorted(unknown)}"
    return True, None


class InMemoryKafkaStub:
    """A drop-in stand-in for a real Kafka topic, used both by tests and by
    the dashboard's 'simulate a claim stream' demo page when no live broker
    is configured. Same producer/consumer call shape as the real client
    libraries (`produce(topic, value)` / iterate to consume), so swapping in
    `confluent_kafka.Producer`/`Consumer` against a real broker is a
    constructor change, not a logic change."""

    def __init__(self):
        self._queue: list[str] = []

    def produce(self, topic: str, value: dict) -> None:
        self._queue.append(json.dumps(value))

    def poll(self, *_args, **_kwargs) -> None:
        return None

    def consume_all(self) -> list[dict]:
        drained, self._queue = self._queue, []
        return [json.loads(m) for m in drained]


def produce_claim(stub: InMemoryKafkaStub, topic: str, claim_payload: dict) -> None:
    ok, error = validate_claim_message(claim_payload)
    if not ok:
        raise ValueError(f"rejected claim message: {error}")
    stub.produce(topic, claim_payload)


def consume_and_process(stub: InMemoryKafkaStub, handler: Callable[[dict], None]) -> int:
    """At-least-once semantics (disclosed, not exactly-once — same as the
    sibling MoMo Guard project's Kafka consumer): a handler failure part-way
    through a batch means already-processed messages ahead of it stay
    processed, and the batch is not retried from the start."""
    messages = stub.consume_all()
    for msg in messages:
        handler(msg)
    return len(messages)
