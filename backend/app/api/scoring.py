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
"""
from __future__ import annotations

import io

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.api.schemas import ClaimIn, ScoreOut
from app.db.models import AuditLogEntry, Claim, ScoredClaim
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

    claim = Claim(external_ref=claim_in.external_ref, raw_payload=payload_dict, ingested_via="api")
    db.add(claim)
    db.flush()
    scored = ScoredClaim(
        claim_id=claim.id, fraud_probability=result["fraud_probability"], risk_grade=result["risk_grade"],
        flagged=result["flagged"], operating_threshold=result["operating_threshold"],
        top_reasons=result["top_reasons"], model_version=result["model_version"],
    )
    db.add(scored)
    db.add(AuditLogEntry(event_type="claim_scored", claim_id=claim.id, detail={"fraud_probability": result["fraud_probability"]}))
    db.commit()

    return ScoreOut(claim_id=claim.id, **result)


@router.post("/score/batch")
async def score_batch(file: UploadFile, db: Session = Depends(get_db)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "upload a .csv file")
    raw = await file.read()
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

    _validate_numeric_columns(df)

    service = FraudScoringService.instance()
    scored = service.score_batch(df)

    claim_ids = []
    for idx, row in df.iterrows():
        claim = Claim(raw_payload=row.to_dict(), ingested_via="batch_csv")
        db.add(claim)
        db.flush()
        s = scored.loc[idx]
        db.add(ScoredClaim(
            claim_id=claim.id, fraud_probability=float(s["fraud_probability"]), risk_grade=s["risk_grade"],
            flagged=bool(s["flagged"]), operating_threshold=float(s["operating_threshold"]),
            model_version=service.model_version,
        ))
        claim_ids.append(claim.id)
    db.add(AuditLogEntry(event_type="batch_scored", detail={"n_rows": len(df)}))
    db.commit()

    scored = scored.reset_index(drop=True)
    scored.insert(0, "claim_id", claim_ids)
    return scored.to_dict(orient="records")
