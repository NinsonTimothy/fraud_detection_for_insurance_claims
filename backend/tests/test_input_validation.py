"""test_input_validation.py — regression coverage for PB-06: ClaimIn.payload
used to be `dict[str, Any]`, so /score silently accepted typo'd/unknown
keys, out-of-range or wrong-typed values, and even a fully empty payload,
scoring all of them as if nothing were wrong. Reproduced directly against
a live TestClient before the fix (see schemas.py's module docstring for
the exact before/after evidence); this file locks in the fixed behavior.
"""
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
    # PB-11: see test_api.py's fixture docstring — every business endpoint
    # now requires X-API-Key.
    monkeypatch.setenv("AEGIS_API_KEY", TEST_API_KEY)
    for mod in list(sys.modules):
        if mod.startswith("app."):
            del sys.modules[mod]
    from app.main import app
    with TestClient(app, headers={"X-API-Key": TEST_API_KEY}) as c:
        yield c


def test_valid_claim_still_scores_successfully(client):
    r = client.post("/score", json={"payload": VALID_CLAIM})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    # every field in VALID_CLAIM was supplied, so none of them should be
    # reported as defaulted.
    assert not (set(VALID_CLAIM.keys()) & set(body["defaulted_fields"]))


def test_unknown_key_is_rejected_not_silently_ignored(client):
    """Before PB-06: a typo'd "aage" was silently ignored and the real
    "age" silently fell back to its default — no error at all."""
    r = client.post("/score", json={"payload": {**VALID_CLAIM, "aage": 35}})
    assert r.status_code == 422
    assert "aage" in r.text


def test_out_of_range_value_is_rejected(client):
    """Before PB-06: age=-999 and a negative claim amount were silently
    accepted and scored as if they were real data."""
    r = client.post("/score", json={"payload": {**VALID_CLAIM, "age": -999}})
    assert r.status_code == 422

    r = client.post("/score", json={"payload": {**VALID_CLAIM, "total_claim_amount": -50000}})
    assert r.status_code == 422


def test_wrong_type_value_is_rejected(client):
    """Before PB-06: age="thirty-five" was silently accepted, then
    feature_engineering's pd.to_numeric(..., errors="coerce").fillna(0.0)
    turned it into age=0 with no error at all — worse than the documented
    default."""
    r = client.post("/score", json={"payload": {**VALID_CLAIM, "age": "thirty-five"}})
    assert r.status_code == 422


def test_invalid_enum_value_is_rejected(client):
    r = client.post("/score", json={"payload": {**VALID_CLAIM, "insured_sex": "OTHER"}})
    assert r.status_code == 422


def test_invalid_date_format_is_rejected(client):
    r = client.post("/score", json={"payload": {**VALID_CLAIM, "incident_date": "05/10/2023"}})
    assert r.status_code == 422


def test_empty_payload_dict_is_rejected(client):
    """Before PB-06: {"payload": {}} was silently accepted and scored — a
    fully-default, meaningless claim with no signal that nothing real was
    submitted. Distinct from a fully-missing body (already covered by
    test_api.py::test_score_rejects_missing_body)."""
    r = client.post("/score", json={"payload": {}})
    assert r.status_code == 422


def test_partial_claim_reports_defaulted_fields(client):
    r = client.post("/score", json={"payload": {"age": 35, "total_claim_amount": 60000.0}})
    assert r.status_code == 200
    body = r.json()
    assert "age" not in body["defaulted_fields"]
    assert "total_claim_amount" not in body["defaulted_fields"]
    assert "insured_sex" in body["defaulted_fields"]
    assert "policy_state" in body["defaulted_fields"]
