"""tests/conftest.py — isolates the dashboard's DB writes into a
throwaway per-session SQLite file instead of the real dev DB
(<project_root>/aegis.db, app/core/config.py's DATABASE_URL default).

PB-12 gave the dashboard its first-ever DB writes (Score a claim / Batch
review now persist via db/persistence.py). Without this, every dashboard
test run would silently accumulate rows in the actual project DB file —
config.DATABASE_URL is read once at import time, and
components/data_access.py's get_db_session_factory() is @st.cache_resource
(a process-global cache), so the env var below MUST be set before
anything imports app.core.config/app.db.session for the first time —
i.e. before any test module (or the page scripts AppTest runs) executes.
conftest.py is imported by pytest before test collection, which is early
enough."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

_tmp_dir = tempfile.mkdtemp(prefix="aegis_dashboard_test_db_")
os.environ["DATABASE_URL"] = f"sqlite:///{Path(_tmp_dir) / 'test_dashboard.db'}"
