"""api/scoring.py — POST /score (single) and /score/batch (CSV upload).

PB-07 (fixed): three expected client mistakes used to 500 instead of
returning a clean 4xx — reproduced directly against a live TestClient
before this fix:
  - POSTing the same external_ref twice to /score raised an unhandled
    sqlalchemy.exc.IntegrityError (UNIQUE constraint failed) -> 500.
  - Uploading a genuinely empty file to /score/batch raised
    pandas.errors.EmptyDataError ("No columns to parse from file") -> 500.
  - Uploading a header-only CSV (0 data rows) got past pd.read_csv but
    then raised sklearn's ValueError ("Found array with 0 sample(s)...")
    deep inside StandardScaler.transform() -> 500.
A fourth related issue — a non-numeric value in a numeric column (e.g.
age="thirty") — did NOT 500: feature_engineering.py's
pd.to_numeric(..., errors="coerce").fillna(0.0) silently turned it into
0.0 with no error and no signal to the caller, the batch-CSV analogue of
the single-claim bug PB-06 fixed. Guarded against here too, as a 422
listing exactly which columns/rows failed to parse, rather than either
crashing or silently corrupting the data.

PB-18 (fixed): /score/batch had three separate problems, reproduced
directly before this fix:
  - N+1 inserts: one `db.flush()` per row inside a Python for-loop, so
    scoring a batch of N claims did N synchronous DB round-trips just to
    assign ids, instead of one. Reproduced by counting actual
    `Session.flush()` calls (tests/test_persistence.py) — N for the old
    per-row loop, 1 for `persist_scored_claims_batch()` regardless of N;
    a linearly-growing cost that matters most against a network-attached
    Postgres (the docker-compose reference deployment), negligible only
    against local SQLite.
  - No `top_reasons` was ever computed or stored for a batch-scored
    claim — score_one()'s SHAP explanation never had a batch
    counterpart, so a claim scored via /score/batch had strictly less
    information than the identical claim scored via /score. Fixed in
    inference.py's score_batch() (explainer.py's new
    top_reasons_batch() — ONE SHAP call over the whole matrix, not N).
  - The uploaded file had no size limit at all — an arbitrarily large
    CSV would be read fully into memory, parsed, scored, and persisted
    in one request. config.MAX_BATCH_UPLOAD_BYTES/MAX_BATCH_ROWS cap
    both the raw upload size and the row count.
"""
from __future__ import annotations

import io

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.api.schemas import ClaimIn, ScoreOut
from app.core import config
from app.db.models import Claim
from app.db.persistence import persist_scored_claim, persist_scored_claims_batch
from app.db.session import get_db
from app.ml.feature_engineering import NUMERIC_PASSTHROUGH_COLUMNS
from app.ml.inference import FraudScoringService

router = APIRouter(tags=["scoring"])


def _validate_numeric_columns(df: pd.DataFrame) -> None:
    """PB-07: raise a clear 422 for any value in a numeric raw column that
    doesn't parse as a number, instead of letting
    feature_engineering.py's pd.to_numeric(errors="coerce").fillna(0.0)
    silently turn it into 0.0."""
    bad_columns: dict[str, list[int]] = {}
    for col in NUMERIC_PASSTHROUGH_COLUMNS:
        if col not in df.columns:
            continue
        raw = df[col].astype(str).str.strip()
        coerced = pd.to_numeric(df[col], errors="coerce")
        bad_mask = coerced.isna() & (raw != "") & (raw.str.lower() != "nan")
        if bad_mask.any():
            bad_columns[col] = df.index[bad_mask].tolist()[:10]
    if bad_columns:
        raise HTTPException(422, detail={
            "error": "non-numeric value(s) in numeric column(s)",
            "columns": bad_columns,
        })


@router.post("/score", response_model=ScoreOut)
def score_claim(claim_in: ClaimIn, db: Session = Depends(get_db)):
    # PB-06: claim_in.payload is now a validated ClaimPayload (typed,
    # ranged, enum-checked, unknown keys rejected — see schemas.py) rather
    # than an untyped dict. by_alias so capital-gains/capital-loss (not
    # valid Python identifiers) round-trip under their real raw-schema
    # names; exclude_none so a field the caller didn't set is genuinely
    # ABSENT from the dict (not present-with-value-None), matching
    # apply_missing_defaults()'s "not supplied" semantics and letting
    # score_one() correctly report it in defaulted_fields.
    payload_dict = claim_in.payload.model_dump(exclude_none=True, by_alias=True)

    # PB-07: check for a duplicate external_ref BEFORE inserting, so a
    # resubmission gets a clean 409 instead of an unhandled IntegrityError.
    if claim_in.external_ref is not None:
        existing = db.query(Claim).filter(Claim.external_ref == claim_in.external_ref).first()
        if existing is not None:
            raise HTTPException(409, f"a claim with external_ref={claim_in.external_ref!r} already exists (claim_id={existing.id})")

    service = FraudScoringService.instance()
    result = service.score_one(payload_dict)

    # PB-12: shared with the dashboard's "Score a claim" page
    # (db/persistence.py) — both surfaces write a Claim/ScoredClaim/audit
    # entry through the exact same function now, not two hand-maintained
    # copies of this logic.
    claim = persist_scored_claim(db, payload_dict, result, external_ref=claim_in.external_ref, ingested_via="api")
    db.commit()

    return ScoreOut(claim_id=claim.id, **result)


@router.post("/score/batch")
async def score_batch(file: UploadFile, db: Session = Depends(get_db)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "upload a .csv file")
    raw = await file.read()

    # PB-18: reject an oversized upload before pandas even parses it —
    # previously unbounded, so an arbitrarily large file was read fully
    # into memory with no check at all until (if ever) something else
    # downstream happened to fail.
    if len(raw) > config.MAX_BATCH_UPLOAD_BYTES:
        raise HTTPException(413, f"upload too large: {len(raw)} bytes (limit {config.MAX_BATCH_UPLOAD_BYTES})")

    # SH-01: keep_default_na=False so an uploaded claim whose
    # authorities_contacted is genuinely "None" (no authority contacted)
    # isn't misread as missing data — see clean_data.py's module docstring.
    try:
        df = pd.read_csv(io.BytesIO(raw), keep_default_na=False, na_values=[""])
    except pd.errors.EmptyDataError:
        raise HTTPException(400, "the uploaded CSV is empty")
    except pd.errors.ParserError as e:
        raise HTTPException(400, f"could not parse the uploaded CSV: {e}")

    if len(df) == 0:
        raise HTTPException(400, "the uploaded CSV has a header but no data rows")

    # PB-18: a small file can still unpack into far more rows than this
    # endpoint should score/persist/explain in one request.
    if len(df) > config.MAX_BATCH_ROWS:
        raise HTTPException(413, f"too many rows: {len(df)} (limit {config.MAX_BATCH_ROWS})")

    _validate_numeric_columns(df)

    service = FraudScoringService.instance()
    scored = service.score_batch(df)

    # PB-18: persist via the shared batch-write helper (db/persistence.py,
    # already used by the dashboard's Batch review page since PB-12) —
    # ONE flush for every Claim row and ONE add_all for every ScoredClaim,
    # instead of a flush per row (previously N synchronous DB round-trips
    # for a batch of N — the cost that matters most against a
    # network-attached Postgres, not local SQLite). score_batch() now
    # also returns a real top_reasons list per row (PB-18), so this
    # persists actual explanations instead of storing NULL for every
    # batch-scored claim.
    raw_rows = df.to_dict(orient="records")
    scored_rows = [{**row, "model_version": service.model_version} for row in scored.to_dict(orient="records")]
    claim_ids = persist_scored_claims_batch(db, raw_rows, scored_rows, ingested_via="batch_csv")
    db.commit()

    scored = scored.reset_index(drop=True)
    scored.insert(0, "claim_id", claim_ids)
    return scored.to_dict(orient="records")
