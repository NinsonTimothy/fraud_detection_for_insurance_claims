"""test_auth.py — regression coverage for PB-11: API-key auth + PII masking.

Reproduced directly before the fix: a fresh, completely unauthenticated
TestClient (no headers at all) could hit every business endpoint,
including `GET /claims`, which returned every stored claim's full
`raw_payload` — including `insured_zip`, a quasi-identifier — verbatim.
"""
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

TEST_API_KEY = "test-suite-api-key"


@pytest.fixture()
def app_module(tmp_path, monkeypatch):
    """Yields the freshly-imported `app.main` module itself (not a
    TestClient) so individual tests can build clients with DIFFERENT
    headers — some authenticated, some not, some with the wrong key."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("AEGIS_API_KEY", TEST_API_KEY)
    for mod in list(sys.modules):
        if mod.startswith("app."):
            del sys.modules[mod]
    import app.main
    return app.main


def test_business_endpoints_reject_missing_api_key(app_module):
    with TestClient(app_module.app) as c:  # no headers at all
        assert c.post("/score", json={"payload": SAMPLE_CLAIM}).status_code == 401
        assert c.get("/claims").status_code == 401
        assert c.get("/claims/risk-grid/summary").status_code == 401
        assert c.get("/audit").status_code == 401
        assert c.get("/monitoring/kpis").status_code == 401


def test_business_endpoints_reject_wrong_api_key(app_module):
    with TestClient(app_module.app, headers={"X-API-Key": "not-the-real-key"}) as c:
        r = c.post("/score", json={"payload": SAMPLE_CLAIM})
        assert r.status_code == 401


def test_business_endpoints_accept_correct_api_key(app_module):
    with TestClient(app_module.app, headers={"X-API-Key": TEST_API_KEY}) as c:
        r = c.post("/score", json={"payload": SAMPLE_CLAIM})
        assert r.status_code == 200


def test_health_and_docs_do_not_require_api_key(app_module):
    """A health check needs to be reachable before any credential is
    provisioned — standard practice, and explicitly NOT a PB-11 gap."""
    with TestClient(app_module.app) as c:  # no headers
        assert c.get("/health").status_code == 200
        assert c.get("/openapi.json").status_code == 200


def test_claims_list_masks_insured_zip(app_module):
    with TestClient(app_module.app, headers={"X-API-Key": TEST_API_KEY}) as c:
        c.post("/score", json={"payload": SAMPLE_CLAIM})
        r = c.get("/claims")
        assert r.status_code == 200
        rows = r.json()
        assert rows, "expected at least one scored claim"
        zip_shown = rows[0]["raw_payload"]["insured_zip"]
        assert zip_shown != SAMPLE_CLAIM["insured_zip"]
        assert zip_shown != str(SAMPLE_CLAIM["insured_zip"])
        assert "X" in zip_shown


def test_get_claim_by_id_masks_insured_zip(app_module):
    with TestClient(app_module.app, headers={"X-API-Key": TEST_API_KEY}) as c:
        r = c.post("/score", json={"payload": SAMPLE_CLAIM})
        claim_id = r.json()["claim_id"]
        r = c.get(f"/claims/{claim_id}")
        assert r.status_code == 200
        assert r.json()["raw_payload"]["insured_zip"] != SAMPLE_CLAIM["insured_zip"]


def test_mask_pii_does_not_mutate_input_dict():
    from app.core.security import mask_pii
    original = {"insured_zip": 468000, "age": 35}
    masked = mask_pii(original)
    assert original["insured_zip"] == 468000  # untouched
    assert masked["insured_zip"] != 468000
    assert masked["age"] == 35  # non-PII fields pass through unchanged


def test_mask_pii_handles_missing_or_none_zip_gracefully():
    from app.core.security import mask_pii
    assert mask_pii({"age": 35}) == {"age": 35}
    assert mask_pii({"insured_zip": None, "age": 35}) == {"insured_zip": None, "age": 35}


def test_require_api_key_uses_constant_time_comparison(monkeypatch):
    """Not a behavioral test so much as a guard against someone
    "simplifying" this back to `provided == API_KEY` later."""
    import inspect

    from app.core import security
    source = inspect.getsource(security.require_api_key)
    assert "compare_digest" in source
