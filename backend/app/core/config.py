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

KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
CLAIMS_TOPIC = os.environ.get("CLAIMS_TOPIC", "aegis.claims.raw")

ANALYST_REVIEW_COST = float(os.environ.get("FP_REVIEW_COST", "250.0"))

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
