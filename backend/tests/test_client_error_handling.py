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


def test_get_nonexistent_claim_is_404(client):
    """PB-16: GET /claims/{id} for an id that was never scored must return
    a clean 404, not a 500 from an unguarded None.score/None.raw_payload
    access downstream."""
    r = client.get("/claims/999999")
    assert r.status_code == 404


def test_feedback_for_nonexistent_claim_is_404_not_409_or_500(client):
    """PB-16: submitting feedback against a claim_id that was never scored
    must 404 (the claim itself doesn't exist) — distinct from the
    already-covered 409 case (the claim exists but already has
    feedback)."""
    r = client.post("/feedback", json={"claim_id": 999999, "investigator_name": "Alice", "confirmed_fraud": True})
    assert r.status_code == 404


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


def test_batch_csv_with_missing_optional_numeric_field_is_200_not_422(client):
    """PB-20: `_validate_numeric_columns` used to false-422 any batch row
    with a genuinely-empty optional numeric cell, because its "present"
    check relied on `.astype(str)` turning a missing cell into the
    literal string "nan" — which this project's pinned pandas (3.0.2) no
    longer does (see `_present_mask`'s docstring in scoring.py). A row
    with total_claim_amount simply absent must score normally, not 422."""
    csv_bytes = io.BytesIO(b"age,total_claim_amount\n35,\n40,60000\n")
    r = client.post("/score/batch", files={"file": ("missing_optional.csv", csv_bytes, "text/csv")})
    assert r.status_code == 200
    assert len(r.json()) == 2


def test_invalid_insured_zip_in_batch_csv_is_422(client):
    """PB-20: /score/batch used to accept ANY value for insured_zip with
    no validation at all, unlike /score's ClaimPayload schema (range
    10,000-999,999). An out-of-range or non-numeric zip must now 422
    with the failing row identified, exactly like /score already does."""
    csv_bytes = io.BytesIO(b"age,insured_zip\n35,99\n40,not-a-zip\n")
    r = client.post("/score/batch", files={"file": ("bad_zip.csv", csv_bytes, "text/csv")})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["columns"]["insured_zip"] == [0, 1]


def test_invalid_incident_date_in_batch_csv_is_422(client):
    """PB-20: a malformed incident_date/policy_bind_date used to sail
    through /score/batch (no equivalent of /score's ISO-format schema
    validation) and silently feed feature_engineering.py's per-batch-
    median NaT fallback. Must now 422 with the offending row identified."""
    csv_bytes = io.BytesIO(b"age,incident_date\n35,2023-05-10\n40,not-a-date\n")
    r = client.post("/score/batch", files={"file": ("bad_date.csv", csv_bytes, "text/csv")})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["columns"]["incident_date"] == [1]


def test_missing_zip_and_dates_in_batch_csv_is_not_an_error(client):
    """PB-20: an ABSENT insured_zip/incident_date/policy_bind_date cell is
    "not supplied", not invalid — must not trip the new 422 validation,
    matching this API's "missing means use the default" semantics
    everywhere else (apply_missing_defaults, etc.)."""
    csv_bytes = io.BytesIO(b"age,insured_zip,incident_date\n35,,\n40,468000,2023-05-10\n")
    r = client.post("/score/batch", files={"file": ("missing_zip_date.csv", csv_bytes, "text/csv")})
    assert r.status_code == 200
    assert len(r.json()) == 2


def test_batch_scored_claim_zip_masks_identically_to_single_scored_claim(client):
    """PB-20: a batch-scored claim's insured_zip used to be corrupted in
    storage by pandas' automatic int->float64 upcast whenever ANY row in
    the batch had a missing zip — reproduced directly as GET
    /claims/{id} showing '46XXXXXX' (8 chars, from the stored float
    468000.0) for the batch-scored claim vs. the correct '46XXXX' (6
    chars) for the identical zip scored via /score, and a genuinely
    missing zip in that same batch round-tripping as the nonsensical
    'naX' instead of a real null. Fixed by `_clean_zip_column` rebuilding
    insured_zip as plain int/None per cell before it's ever persisted."""
    single = client.post("/score", json={"payload": {**VALID_CLAIM, "insured_zip": 468000}})
    assert single.status_code == 200
    single_detail = client.get(f"/claims/{single.json()['claim_id']}")
    single_masked_zip = single_detail.json()["raw_payload"]["insured_zip"]
    assert single_masked_zip == "46XXXX"

    # One row shares the same zip as the single-claim submission above;
    # the other row's zip is genuinely missing — deliberately mixed in
    # the same batch, since the original bug was triggered by pandas
    # upcasting the WHOLE column the instant any one cell was empty.
    csv_bytes = io.BytesIO(b"age,insured_zip\n35,468000\n40,\n")
    batch = client.post("/score/batch", files={"file": ("zips.csv", csv_bytes, "text/csv")})
    assert batch.status_code == 200
    rows = batch.json()

    present_row = client.get(f"/claims/{rows[0]['claim_id']}").json()
    assert present_row["raw_payload"]["insured_zip"] == single_masked_zip == "46XXXX"

    missing_row = client.get(f"/claims/{rows[1]['claim_id']}").json()
    assert missing_row["raw_payload"]["insured_zip"] is None
