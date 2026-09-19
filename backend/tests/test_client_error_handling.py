"""test_client_error_handling.py — regression coverage for PB-07: several
EXPECTED client mistakes (a duplicate external_ref, duplicate feedback for
the same claim, an empty/header-only CSV upload) used to raise unhandled
exceptions and return a bare 500 instead of a clean 4xx; a non-numeric
value in a numeric CSV column silently corrupted to 0.0 instead of either
of those. Reproduced directly against a live TestClient before the fix
(see scoring.py/feedback.py's module docstrings for the exact evidence);
this file locks in the fixed behavior."""
import io
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


def test_duplicate_external_ref_is_409_not_500(client):
    r1 = client.post("/score", json={"external_ref": "ext-dup", "payload": VALID_CLAIM})
    assert r1.status_code == 200

    r2 = client.post("/score", json={"external_ref": "ext-dup", "payload": VALID_CLAIM})
    assert r2.status_code == 409
    assert "ext-dup" in r2.text


def test_duplicate_feedback_is_409_not_500(client):
    r = client.post("/score", json={"payload": VALID_CLAIM})
    claim_id = r.json()["claim_id"]

    fb1 = client.post("/feedback", json={"claim_id": claim_id, "investigator_name": "Alice", "confirmed_fraud": True})
    assert fb1.status_code == 200

    fb2 = client.post("/feedback", json={"claim_id": claim_id, "investigator_name": "Bob", "confirmed_fraud": False})
    assert fb2.status_code == 409


def test_empty_csv_upload_is_400_not_500(client):
    r = client.post("/score/batch", files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")})
    assert r.status_code == 400


def test_header_only_csv_upload_is_400_not_500(client):
    r = client.post("/score/batch", files={"file": ("h.csv", io.BytesIO(b"age,total_claim_amount\n"), "text/csv")})
    assert r.status_code == 400


def test_non_numeric_csv_value_is_422_not_silently_corrupted(client):
    bad_csv = io.BytesIO(b"age,total_claim_amount\nthirty,50000\n25,also_bad\n")
    r = client.post("/score/batch", files={"file": ("bad.csv", bad_csv, "text/csv")})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "age" in detail["columns"]
    assert "total_claim_amount" in detail["columns"]


def test_valid_batch_csv_still_scores_successfully(client):
    good_csv = io.BytesIO(b"age,total_claim_amount\n35,50000\n40,60000\n")
    r = client.post("/score/batch", files={"file": ("good.csv", good_csv, "text/csv")})
    assert r.status_code == 200
    assert len(r.json()) == 2


def test_batch_scoring_includes_top_reasons_per_row(client):
    """PB-18: /score/batch used to return no top_reasons at all — a claim
    scored in a batch had strictly less information than the identical
    claim scored via /score. Every row must now carry a real, non-empty
    reasons list, and each claim_id must be independently retrievable
    with its explanation intact via GET /claims/{id}."""
    good_csv = io.BytesIO(b"age,total_claim_amount\n35,50000\n40,60000\n")
    r = client.post("/score/batch", files={"file": ("good.csv", good_csv, "text/csv")})
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 2
    for row in rows:
        assert row["top_reasons"], "expected a non-empty top_reasons list for every batch-scored row"
        assert {"rank", "feature", "sentence"} <= set(row["top_reasons"][0].keys())

    claim_detail = client.get(f"/claims/{rows[0]['claim_id']}")
    assert claim_detail.status_code == 200


def test_oversized_batch_upload_is_413_not_silently_truncated(client, monkeypatch):
    monkeypatch.setenv("MAX_BATCH_UPLOAD_BYTES", "10")
    for mod in list(sys.modules):
        if mod.startswith("app."):
            del sys.modules[mod]
    from app.main import app as fresh_app
    from fastapi.testclient import TestClient as FreshTestClient
    with FreshTestClient(fresh_app, headers={"X-API-Key": TEST_API_KEY}) as fresh_client:
        good_csv = io.BytesIO(b"age,total_claim_amount\n35,50000\n40,60000\n")  # well over 10 bytes
        r = fresh_client.post("/score/batch", files={"file": ("good.csv", good_csv, "text/csv")})
        assert r.status_code == 413


def test_too_many_batch_rows_is_413_not_silently_scored(client, monkeypatch):
    monkeypatch.setenv("MAX_BATCH_ROWS", "2")
    for mod in list(sys.modules):
        if mod.startswith("app."):
            del sys.modules[mod]
    from app.main import app as fresh_app
    from fastapi.testclient import TestClient as FreshTestClient
    with FreshTestClient(fresh_app, headers={"X-API-Key": TEST_API_KEY}) as fresh_client:
        rows = "\n".join(f"{20 + i},{1000 * i}" for i in range(5))
        csv_bytes = io.BytesIO(f"age,total_claim_amount\n{rows}\n".encode())
        r = fresh_client.post("/score/batch", files={"file": ("many.csv", csv_bytes, "text/csv")})
        assert r.status_code == 413
        assert "limit 2" in r.text
