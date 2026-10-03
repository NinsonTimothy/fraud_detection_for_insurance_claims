"""test_deployment_config.py — regression coverage for PB-13 (Docker/
deployment config: D2 SQLite-default/Postgres-optional, credentials,
ports, train-init).

Uses `docker compose config --format json` (stdlib json only — no PyYAML
dependency) against the real docker-compose.yml, so these tests catch a
regression in the actual file, not a copy of it. Skips cleanly wherever
the `docker` CLI isn't available (this repo has no hard runtime
dependency on Docker itself — see docs/LIMITATIONS.md's Docker section)
rather than failing the whole suite in an environment that doesn't have
it; CI (`.github/workflows/tests.yml`, ubuntu-latest) does.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(shutil.which("docker") is None, reason="docker CLI not available in this environment")


def _compose_config(env: dict | None = None) -> dict:
    import os
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    result = subprocess.run(
        ["docker", "compose", "-f", str(REPO_ROOT / "docker-compose.yml"), "config", "--format", "json"],
        cwd=REPO_ROOT, capture_output=True, text=True, env=full_env, timeout=30,
    )
    assert result.returncode == 0, f"docker compose config failed:\n{result.stderr}"
    return json.loads(result.stdout)


def test_compose_file_is_syntactically_valid():
    config = _compose_config()
    assert set(config["services"]) == {"postgres", "api", "dashboard", "train-init"}


def test_postgres_port_is_not_published_to_host():
    """PB-13: 5432 used to be published to the host ("5432:5432") for a
    service every consumer already reaches over the compose-internal
    network by name — needless exposure of a DB with default aegis/aegis
    credentials. `postgres` should have no `ports` entry at all."""
    config = _compose_config()
    assert "ports" not in config["services"]["postgres"] or not config["services"]["postgres"]["ports"]


def test_api_and_dashboard_ports_are_still_published():
    """Making sure the postgres-port fix didn't also strip the ports
    those two services genuinely need to be reachable from the host."""
    config = _compose_config()
    api_ports = {p["target"] for p in config["services"]["api"]["ports"]}
    dashboard_ports = {p["target"] for p in config["services"]["dashboard"]["ports"]}
    assert 8000 in api_ports
    assert 8501 in dashboard_ports


def test_train_init_has_no_postgres_dependency():
    """PB-13: clean_data.py/train.py/evaluate_oracle.py never open a DB
    connection (grep confirms zero references to db/DATABASE_URL/
    SessionLocal/get_db across all three), so train-init shouldn't wait
    on Postgres's healthcheck or carry a DATABASE_URL it never reads."""
    config = _compose_config()
    train_init = config["services"]["train-init"]
    assert "postgres" not in (train_init.get("depends_on") or {})
    assert "DATABASE_URL" not in (train_init.get("environment") or {})


def test_train_init_modules_never_touch_the_db():
    """The claim the previous test's docstring relies on, checked
    directly against source rather than just asserted in a comment."""
    ml_dir = REPO_ROOT / "backend" / "app" / "ml"
    needles = ("SessionLocal", "get_db", "DATABASE_URL", "app.db")
    for module in ("clean_data.py", "train.py", "evaluate_oracle.py"):
        text = (ml_dir / module).read_text()
        for needle in needles:
            assert needle not in text, f"{module} references {needle} — train-init's no-DB-dependency assumption is stale"


def test_database_url_is_built_from_shared_postgres_credentials_by_default():
    config = _compose_config()
    expected = "postgresql://aegis:aegis@postgres/aegis"
    assert config["services"]["api"]["environment"]["DATABASE_URL"] == expected
    assert config["services"]["dashboard"]["environment"]["DATABASE_URL"] == expected


def test_postgres_credentials_cannot_drift_apart():
    """PB-13: the same POSTGRES_PASSWORD override must land identically
    in the postgres service's own env AND in api/dashboard's DATABASE_URL
    — regression against going back to four hand-typed copies of
    "aegis:aegis" that could silently disagree."""
    config = _compose_config(env={"POSTGRES_USER": "customuser", "POSTGRES_PASSWORD": "custompass", "POSTGRES_DB": "customdb"})
    pg_env = config["services"]["postgres"]["environment"]
    assert pg_env["POSTGRES_USER"] == "customuser"
    assert pg_env["POSTGRES_PASSWORD"] == "custompass"
    assert pg_env["POSTGRES_DB"] == "customdb"

    expected_url = "postgresql://customuser:custompass@postgres/customdb"
    assert config["services"]["api"]["environment"]["DATABASE_URL"] == expected_url
    assert config["services"]["dashboard"]["environment"]["DATABASE_URL"] == expected_url


def test_database_url_can_be_fully_overridden_to_skip_postgres():
    """D2: Postgres is the reference datastore, never a hard requirement
    — a full DATABASE_URL override (e.g. sqlite:///) must take precedence
    over the POSTGRES_* composition."""
    config = _compose_config(env={"DATABASE_URL": "sqlite:////srv/data/aegis.db"})
    assert config["services"]["api"]["environment"]["DATABASE_URL"] == "sqlite:////srv/data/aegis.db"
    assert config["services"]["dashboard"]["environment"]["DATABASE_URL"] == "sqlite:////srv/data/aegis.db"


def test_env_example_documents_every_overridable_variable():
    env_example = (REPO_ROOT / ".env.example").read_text()
    for var in ("AEGIS_API_KEY", "POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB", "DATABASE_URL"):
        assert var in env_example, f"{var} is settable in docker-compose.yml but undocumented in .env.example"
