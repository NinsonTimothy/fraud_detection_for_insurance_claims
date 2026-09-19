"""db/session.py — engine/session factory. `connect_args` is SQLite-only;
Postgres in docker-compose ignores it via the URL scheme check below."""
from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import DATABASE_URL
from app.db.models import Base

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    # Prototype-only schema management (PB-22): create_all() creates any
    # missing table but never ALTERs an existing one, so a model field
    # added/renamed/removed after first run needs a manual DB reset (or a
    # migration tool such as Alembic) — there is no migration history here.
    # Fine for a thesis-scale SQLite/single-environment deployment; not a
    # substitute for real migrations in a long-lived, multi-environment one.
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
