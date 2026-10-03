"""
core/security.py — API-key auth + PII masking (PB-11).

PB-11 (fixed): every endpoint (scoring, claims lookup, feedback, audit,
monitoring) was reachable with no authentication at all, and
`GET /claims`/`GET /claims/{id}` returned a claim's full `raw_payload`
verbatim — including `insured_zip`, a quasi-identifier (HIPAA's Safe
Harbor rule treats a full ZIP the same way, alongside age/sex, which this
payload also carries) that doesn't need to leave the system in the clear
for any of this API's actual use cases (scoring itself only needs it
in-process, never in a response body). Reproduced directly: a fresh
`TestClient` with no headers at all could `GET /claims` and read back
every stored claim's raw ZIP.

Fix: a single shared-secret API key, checked via FastAPI's own
`APIKeyHeader` (so it shows up as a proper "Authorize" button in
`/docs`, not just an undocumented header a caller has to already know
about), applied to every business router in `main.py` — `/health` and
the auto-generated docs routes stay open, matching standard practice
(a health check needs to be reachable for infrastructure monitoring
before any credential is provisioned). `mask_pii()` redacts
`insured_zip` in every response that echoes back a claim's raw payload;
scoring itself is untouched, since it needs the real value.

This is a single, deployment-wide static key (`AEGIS_API_KEY`), not
per-user auth/RBAC — genuinely out of scope for what this project's own
`docs/LIMITATIONS.md` already discloses under "Minimal PII handling" and
"no field-level encryption or masking." It closes the "wide open, no
credential needed at all" gap without pretending to be a full identity
system.
"""
from __future__ import annotations

import hmac

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from app.core.config import API_KEY

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(provided: str | None = Security(_api_key_header)) -> str:
    """FastAPI dependency — raise 401 if the caller didn't send a matching
    `X-API-Key` header. `hmac.compare_digest` avoids a timing side-channel
    on the comparison (a real, if minor, thing to get right even for a
    single shared secret)."""
    if provided is None or not hmac.compare_digest(provided, API_KEY):
        raise HTTPException(401, "missing or invalid X-API-Key header")
    return provided


# PB-11: fields that must never leave this API unmasked in a response body
# — currently just insured_zip (a quasi-identifier per HIPAA Safe Harbor,
# combined with age/sex which this payload also carries). Scoring itself
# reads the REAL value from the request payload directly; this only
# affects what gets echoed BACK in a claim-lookup response.
PII_FIELDS = ("insured_zip",)


def _mask_zip(value) -> str:
    s = str(value)
    return s[:2] + "X" * max(0, len(s) - 2) if len(s) > 2 else "X" * len(s)


def mask_pii(raw_payload: dict) -> dict:
    """Returns a COPY of `raw_payload` with PII_FIELDS redacted — never
    mutates the original (callers pass the DB-loaded dict; mutating it
    could leak into whatever the ORM session does with it next)."""
    if not isinstance(raw_payload, dict):
        return raw_payload
    masked = dict(raw_payload)
    if "insured_zip" in masked and masked["insured_zip"] is not None:
        masked["insured_zip"] = _mask_zip(masked["insured_zip"])
    return masked
