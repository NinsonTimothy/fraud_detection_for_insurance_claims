"""api/schemas.py — pydantic request/response models.

PB-06 (fixed): `ClaimIn.payload` used to be `dict[str, Any]` — no type
checking, no range checking, no enum checking, and no rejection of
unknown keys. Reproduced directly against a live TestClient:

  - A typo'd key (`"aage": 35` instead of `"age"`) was silently accepted;
    the real `age` silently fell back to its documented default (39) with
    no error and no signal to the caller that their field name was wrong.
  - Garbage values (`"age": -999`, `"total_claim_amount": -50000`) were
    silently accepted and scored as if they were real data.
  - A completely empty `{"payload": {}}` was silently accepted and scored
    (a fully-default, meaningless score) — status 200, no warning.
  - Worst: `{"age": "thirty-five"}` (wrong TYPE entirely) was silently
    accepted. `feature_engineering.py`'s numeric-passthrough columns go
    through `pd.to_numeric(..., errors="coerce").fillna(0.0)`, so an
    unparseable string silently became age=0 — not even the documented
    default, a different and worse silent failure.

Fix: `ClaimPayload` below gives every one of `feature_engineering.py`'s
`RAW_FEATURE_COLUMNS` an explicit type, a range (`Field(ge=..., le=...)`)
or an enum (`Literal[...]`, built from the actual distinct values in
`data/cleaned/insurance_claims_cleaned.csv` for genuinely closed-vocabulary
columns), `model_config = ConfigDict(extra="forbid")` rejects any key
that isn't a real raw field (catches typos immediately, as a 422 instead
of a silent default), and a model validator rejects a payload with no
fields set at all. Every field stays Optional — this project's own design
(the live API, batch CSV upload, and `oracle_adapter.py`'s external
adapter) depends on being able to supply any SUBSET of raw fields and let
`apply_missing_defaults()` fill in the rest, so "required" here means
"correct if present", not "every field must be supplied". Free-text
fields the model only reduces to a single flag (`insured_hobbies` ->
`is_highrisk_hobby`, `insured_occupation` -> `is_exec_occupation`) and
`auto_make` (unused by the model at all — see feature_engineering.py's
CATEGORICAL_COLUMNS note) are intentionally left as free strings rather
than enums, since an unrecognized value there degrades gracefully (it's
just "not chess", "not exec-managerial") rather than silently corrupting
a numeric feature.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _validate_iso_date(v: str | None) -> str | None:
    if v is None:
        return v
    try:
        date.fromisoformat(v)
    except ValueError:
        raise ValueError(f"expected an ISO date (YYYY-MM-DD), got {v!r}")
    return v


class ClaimPayload(BaseModel):
    """One raw claim, matching `feature_engineering.RAW_FEATURE_COLUMNS`.
    Every field is optional (see module docstring); when present, it must
    satisfy its type/range/enum. Unknown keys are rejected outright."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    months_as_customer: int | None = Field(None, ge=0, le=1000)
    age: int | None = Field(None, ge=16, le=100)
    policy_bind_date: str | None = None
    policy_state: Literal["IL", "IN", "OH"] | None = None
    policy_csl: Literal["100/300", "250/500", "500/1000"] | None = None
    policy_deductable: Literal[500, 1000, 2000] | None = None
    policy_annual_premium: float | None = Field(None, ge=0, le=10_000)
    umbrella_limit: int | None = Field(None, ge=-2_000_000, le=20_000_000)
    insured_zip: int | None = Field(None, ge=10_000, le=999_999)
    insured_sex: Literal["MALE", "FEMALE"] | None = None
    insured_education_level: Literal["JD", "High School", "Associate", "College", "Masters", "PhD", "MD"] | None = None
    insured_occupation: str | None = Field(None, max_length=64)
    insured_hobbies: str | None = Field(None, max_length=64)
    insured_relationship: Literal["husband", "wife", "own-child", "other-relative", "not-in-family", "unmarried"] | None = None
    capital_gains: int | None = Field(None, ge=0, le=200_000, alias="capital-gains")
    capital_loss: int | None = Field(None, ge=-200_000, le=0, alias="capital-loss")
    incident_date: str | None = None
    incident_type: Literal["Multi-vehicle Collision", "Single Vehicle Collision", "Vehicle Theft", "Parked Car"] | None = None
    collision_type: Literal["Front Collision", "Rear Collision", "Side Collision"] | None = None
    incident_severity: Literal["Trivial Damage", "Minor Damage", "Major Damage", "Total Loss"] | None = None
    authorities_contacted: Literal["Police", "Fire", "Ambulance", "Other", "None"] | None = None
    incident_state: Literal["NC", "NY", "OH", "PA", "SC", "VA", "WV"] | None = None
    incident_hour_of_the_day: int | None = Field(None, ge=0, le=23)
    number_of_vehicles_involved: int | None = Field(None, ge=1, le=10)
    property_damage: Literal["YES", "NO"] | None = None
    bodily_injuries: int | None = Field(None, ge=0, le=10)
    witnesses: int | None = Field(None, ge=0, le=20)
    police_report_available: Literal["YES", "NO"] | None = None
    total_claim_amount: float | None = Field(None, ge=0, le=1_000_000)
    injury_claim: float | None = Field(None, ge=0, le=1_000_000)
    property_claim: float | None = Field(None, ge=0, le=1_000_000)
    vehicle_claim: float | None = Field(None, ge=0, le=1_000_000)
    auto_make: str | None = Field(None, max_length=32)
    auto_year: int | None = Field(None, ge=1900, le=2030)

    _validate_policy_bind_date = field_validator("policy_bind_date")(_validate_iso_date)
    _validate_incident_date = field_validator("incident_date")(_validate_iso_date)

    @model_validator(mode="after")
    def _at_least_one_field_supplied(self):
        if all(v is None for v in self.model_dump().values()):
            raise ValueError(
                "payload has no fields set — an all-default claim (every "
                "raw field falling back to MISSING_COLUMN_DEFAULTS) is not "
                "a meaningful scoring request; supply at least one real field"
            )
        return self

    def supplied_fields(self) -> set[str]:
        """Field names (by alias, matching RAW_FEATURE_COLUMNS) this
        payload actually set — used to report which raw fields were
        defaulted, not supplied by the caller (PB-06)."""
        return {
            (self.model_fields[name].alias or name): value
            for name, value in self.model_dump(exclude_none=True).items()
        }.keys()


class ClaimIn(BaseModel):
    external_ref: str | None = None
    payload: ClaimPayload = Field(..., description="Raw claim fields — see feature_engineering.RAW_FEATURE_COLUMNS. Missing fields fall back to documented defaults; unknown keys are rejected.")


class ReasonCode(BaseModel):
    # PB-05: rank/display_name/direction/impact are new — a genuinely
    # plain-language reason, not just a feature name + a raw signed SHAP
    # float (see explainer.py's ClaimExplainer.top_reasons() docstring).
    rank: int
    feature: str
    display_name: str
    value: Any
    shap_value: float
    direction: str
    impact: str
    sentence: str


class ScoreOut(BaseModel):
    claim_id: int | None = None
    fraud_probability: float
    risk_grade: str
    flagged: bool
    operating_threshold: float
    recommended_action: str
    defaulted_fields: list[str] = []
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
