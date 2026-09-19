"""test_api.py — exercises the FastAPI app end-to-end via TestClient
against a fresh SQLite DB (no live Postgres needed for these tests)."""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SAMPLE_CLAIM = {
    "months_as_customer": 120, "age": 35, "policy_bind_date": "2018-01-01",
    "policy_deductable": 1000, "policy_annual_premium": 1300.0,
    "insured_zip": 468000, "insured_sex": "MALE", "insured_hobbies": "reading",
    "incident_date": "2023-05-10", "incident_severity": "Major Damage",
    "witnesses": 0, "police_report_available": "YES",
    "total_claim_amount": 60000.0, "injury_claim": 10000.0,
    "property_claim": 15000.0, "vehicle_claim": 35000.0, "auto_year": 2015,
}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    # Re-import fresh so the app picks up the patched DATABASE_URL.
    for mod in list(sys.modules):
        if mod.startswith("app."):
            del sys.modules[mod]
    from app.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_score_and_retrieve_claim(client):
    r = client.post("/score", json={"payload": SAMPLE_CLAIM})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["risk_grade"] in {"Low", "Medium", "High"}
    # PB-05: top_reasons defaults to k=8 (was 3) — see explainer.py's
    # DEFAULT_TOP_K.
    assert len(body["top_reasons"]) == 8
    first = body["top_reasons"][0]
    assert first["rank"] == 1
    assert first["direction"] in {"increased", "decreased"}
    assert first["impact"] in {"strongly", "moderately", "slightly"}
    assert first["display_name"]

    claim_id = body["claim_id"]
    r = client.get(f"/claims/{claim_id}")
    assert r.status_code == 200
    assert r.json()["score"]["fraud_probability"] == pytest.approx(body["fraud_probability"])


def test_feedback_roundtrip_and_export(client):
    r = client.post("/score", json={"payload": SAMPLE_CLAIM})
    claim_id = r.json()["claim_id"]

    r = client.post("/feedback", json={
        "claim_id": claim_id, "investigator_name": "Test Investigator",
        "confirmed_fraud": True, "notes": "verified via callback",
    })
    assert r.status_code == 200

    r = client.get("/feedback/export")
    assert r.status_code == 200
    assert "Test Investigator" in r.text


def test_audit_log_records_scoring_event(client):
    client.post("/score", json={"payload": SAMPLE_CLAIM})
    r = client.get("/audit")
    assert r.status_code == 200
    events = r.json()
    assert any(e["event_type"] == "claim_scored" for e in events)


def test_risk_grid_summary(client):
    client.post("/score", json={"payload": SAMPLE_CLAIM})
    r = client.get("/claims/risk-grid/summary")
    assert r.status_code == 200
    body = r.json()
    assert body["total_scored"] >= 1


def test_kafka_ingest_flow(client):
    r = client.post("/ingest/kafka/produce", json=SAMPLE_CLAIM)
    assert r.status_code == 200
    r = client.post("/ingest/kafka/consume")
    assert r.status_code == 200
    assert r.json()["n_processed"] >= 1


def test_score_rejects_missing_body(client):
    r = client.post("/score", json={})
    assert r.status_code == 422
