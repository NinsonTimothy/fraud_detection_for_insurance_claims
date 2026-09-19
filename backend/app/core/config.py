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
