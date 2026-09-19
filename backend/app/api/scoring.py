"""api/scoring.py — POST /score (single) and /score/batch (CSV upload)."""
from __future__ import annotations

import io

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.api.schemas import ClaimIn, ScoreOut
from app.db.models import AuditLogEntry, Claim, ScoredClaim
from app.db.session import get_db
from app.ml.inference import FraudScoringService

router = APIRouter(tags=["scoring"])


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
    df = pd.read_csv(io.BytesIO(raw), keep_default_na=False, na_values=[""])
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
