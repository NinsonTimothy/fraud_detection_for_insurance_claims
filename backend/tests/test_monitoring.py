"""test_monitoring.py — PB-16: api/monitoring.py had zero behavioral test
coverage before this ticket (test_auth.py only checks that /monitoring/kpis
401s without a key — never that either endpoint actually works or handles
its documented edge case correctly)."""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

VALID_CLAIM = {
    "months_as_customer": 120, "age": 35, "policy_bind_date": "2018-01-01",
    "policy_deductable": 1000, "policy_annual_premium": 1300.0,
    "insured_zip": 468000, "insured_sex": "MALE", "insured_hobbies": "reading",
    "incident_date": "2023-05-10", "incident_severity": "Major Damage",
    "witnesses": 0, "police_report_available": "YES",
    "total_claim_amount": 60000.0, "injury_claim": 10000.0,
    "property_claim": 15000.0, "vehicle_claim": 35000.0, "auto_year": 2015,
}

TEST_API_KEY = "test-suite-api-key"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("AEGIS_API_KEY", TEST_API_KEY)
    for mod in list(sys.modules):
        if mod.startswith("app."):
            del sys.modules[mod]
    from app.main import app
    with TestClient(app, headers={"X-API-Key": TEST_API_KEY}) as c:
        yield c


def test_kpis_reports_real_model_metrics_and_live_counts(client):
    client.post("/score", json={"payload": VALID_CLAIM})
    r = client.get("/monitoring/kpis")
    assert r.status_code == 200
    body = r.json()
    assert "internal_holdout" in body
    assert "operating_threshold" in body
    assert body["total_scored_this_deployment"] >= 1


def test_drift_reports_insufficient_volume_below_the_documented_minimum(client):
    """monitoring.py's own docstring documents a hard minimum of 30 live
    scored claims before it attempts a PSI comparison — below that, it
    must return the documented insufficient_live_volume status instead
    of feeding a tiny/degenerate sample into psi_report() and returning a
    misleadingly precise (and statistically meaningless) drift number."""
    client.post("/score", json={"payload": VALID_CLAIM})  # well under 30
    r = client.get("/monitoring/drift")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "insufficient_live_volume"
    assert body["n_live"] == 1
    assert body["minimum_required"] == 30


def test_drift_returns_a_real_psi_report_once_minimum_volume_is_reached(client):
    for _ in range(30):
        r = client.post("/score", json={"payload": VALID_CLAIM})
        assert r.status_code == 200
    r = client.get("/monitoring/drift")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    assert len(body) == 1
    assert body[0]["feature"] == "y_proba"
    assert body[0]["psi"] >= 0.0
    assert "significant_drift" in body[0]
