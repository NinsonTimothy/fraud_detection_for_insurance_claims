"""api/schemas.py — pydantic request/response models."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ClaimIn(BaseModel):
    external_ref: str | None = None
    payload: dict[str, Any] = Field(..., description="Raw claim fields — see feature_engineering.RAW_FEATURE_COLUMNS. Missing fields fall back to documented defaults.")


class ReasonCode(BaseModel):
    feature: str
    value: Any
    shap_value: float
    sentence: str


class ScoreOut(BaseModel):
    claim_id: int | None = None
    fraud_probability: float
    risk_grade: str
    flagged: bool
    operating_threshold: float
    recommended_action: str
    top_reasons: list[ReasonCode] = []
    model_version: str


class FeedbackIn(BaseModel):
    claim_id: int
    investigator_name: str
    confirmed_fraud: bool
    notes: str | None = None


class BatchScoreRow(BaseModel):
    row_index: int
    fraud_probability: float
    risk_grade: str
    flagged: bool
    recommended_action: str
