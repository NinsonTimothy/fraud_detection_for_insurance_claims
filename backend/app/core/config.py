"""core/config.py — central settings, read once at import time."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# Postgres in docker-compose (service name `postgres`); SQLite fallback for
# local/sandboxed runs where no Postgres server is reachable — same
# tables/schema either way via SQLAlchemy, so nothing else in the app needs
# to know which one is live.
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{PROJECT_ROOT / 'aegis.db'}")

ANALYST_REVIEW_COST = float(os.environ.get("FP_REVIEW_COST", "250.0"))

# PB-11: a single shared-secret API key, checked by core/security.py's
# require_api_key() dependency on every business router. The fallback
# value is INTENTIONALLY an obvious, documented placeholder, not a
# generated secret — this makes "nobody set AEGIS_API_KEY" a visible,
# grep-able fact about a deployment rather than a silently-working
# default that looks secure but isn't. Any real deployment MUST set
# AEGIS_API_KEY — docker-compose.yml's api service reads it from the
# shell/.env.example with this exact same fallback (PB-13), so an
# unconfigured Docker deployment is visibly insecure the same way an
# unconfigured bare `uvicorn` one is, not silently fine.
API_KEY = os.environ.get("AEGIS_API_KEY", "CHANGE-ME-insecure-default-api-key")

# PB-18: /score/batch used to accept an unbounded CSV upload — no cap on
# the raw upload size or on how many rows it unpacked to. Rejecting a
# too-large upload before pandas parses it (MAX_BATCH_UPLOAD_BYTES) and
# capping row count (MAX_BATCH_ROWS) bounds parsing time, scoring/SHAP
# time, DB writes, and response payload size to something finite.
# Defaults are generous for this thesis-scale dataset's shape (the raw
# dataset itself is ~1,000 rows) but not unlimited, and overridable.
MAX_BATCH_UPLOAD_BYTES = int(os.environ.get("MAX_BATCH_UPLOAD_BYTES", str(10 * 1024 * 1024)))  # 10 MB
MAX_BATCH_ROWS = int(os.environ.get("MAX_BATCH_ROWS", "5000"))

# SH-02 / D3: `is_highrisk_hobby` and `is_exec_occupation` (feature_engineering.py's
# RISKY_FEATURE_COLUMNS) encode a lifestyle/occupation -> risk association
# with no causal fraud mechanism behind it (see that file's module
# docstring) — close to proxy-discrimination, the kind real insurance
# regulators scrutinize insurers for. D3's default is OFF: the deployable
# headline model does NOT use them. Set INCLUDE_PROXY_FEATURES=true to
# opt back in (e.g. to reproduce the WITH-proxy comparison numbers in
# docs/REBUILD_NOTES.md / data/processed/proxy_feature_ablation.csv) —
# never the default for anything actually shipped.
INCLUDE_PROXY_FEATURES = os.environ.get("INCLUDE_PROXY_FEATURES", "false").strip().lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Pre-defence round 2 — every methodological assumption lives here, is
# disclosed in the generated docs, and is overridable by environment variable.
# ---------------------------------------------------------------------------
def _floats(name: str, default: str) -> tuple[float, ...]:
    return tuple(float(x) for x in os.environ.get(name, default).split(","))

# Model selection (B3): repeated stratified CV on the 800 development rows.
CV_SPLITS = int(os.environ.get("AEGIS_CV_SPLITS", "5"))
CV_REPEATS = int(os.environ.get("AEGIS_CV_REPEATS", "10"))
SIGNIFICANCE_ALPHA = float(os.environ.get("AEGIS_ALPHA", "0.05"))

# Calibration (B4): a calibration method is adopted only if it lowers the
# development out-of-fold Brier score by at least this much.
CALIBRATION_MIN_BRIER_GAIN = float(os.environ.get("AEGIS_CAL_MIN_GAIN", "0.002"))
ECE_BINS = 10

# Risk bands (B5): derived from development out-of-fold scores.
# High  = the top-scored claims that together contain HIGH_BAND_FRAUD_CAPTURE of all fraud.
# Medium = the next claims down until MEDIUM_BAND_FRAUD_CAPTURE of all fraud is contained.
HIGH_BAND_FRAUD_CAPTURE = float(os.environ.get("AEGIS_HIGH_CAPTURE", "0.50"))
MEDIUM_BAND_FRAUD_CAPTURE = float(os.environ.get("AEGIS_MEDIUM_CAPTURE", "0.85"))

# Cost sensitivity analysis (B6). Expected cost at threshold t =
#   sum over MISSED frauds of (claim amount x fraudulent share x recovery rate)
#   + review cost x (number of claims flagged, true AND false positives).
# Reported as a grid; never used as the operating threshold.
COST_REVIEW_COSTS = _floats("AEGIS_COST_REVIEW", "250,1000,2500,5000")
COST_FRAUD_SHARES = _floats("AEGIS_COST_FRAUD_SHARE", "0.1,0.25,0.5")
COST_RECOVERY_RATES = _floats("AEGIS_COST_RECOVERY", "0.3,0.6,0.9")
