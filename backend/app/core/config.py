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
