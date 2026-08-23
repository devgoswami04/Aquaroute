"""Database engine/session helpers.

All DB access goes through here. ``postgis_available()`` lets callers degrade
gracefully to file-based caches when Docker/Postgres isn't running, so the
pipeline and API stay demonstrable without the container.
"""
from __future__ import annotations

import functools
import socket
from urllib.parse import urlparse

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from aquaroute.config import get_settings


@functools.lru_cache(maxsize=1)
def get_engine() -> Engine:
    url = get_settings().database_url
    return create_engine(url, pool_pre_ping=True, future=True)


@functools.lru_cache(maxsize=1)
def get_sessionmaker():
    return sessionmaker(bind=get_engine(), future=True)


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    """Fast TCP pre-check so we never block on a dead Postgres (Windows can hang
    the psycopg connect for a long time otherwise)."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def postgis_available() -> bool:
    """True if we can connect and run a query. Socket pre-check keeps it instant
    when Postgres isn't running."""
    url = urlparse(get_settings().database_url.replace("postgresql+psycopg", "postgresql"))
    host, port = url.hostname or "localhost", url.port or 5432
    if not _port_open(host, port):
        return False
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def init_db() -> None:
    """Enable PostGIS and create all tables (idempotent)."""
    from aquaroute.db.models import Base

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
    Base.metadata.create_all(engine)
