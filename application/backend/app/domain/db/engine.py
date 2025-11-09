# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Generator
from functools import lru_cache
from pathlib import Path
from sqlite3 import Connection
from typing import Any

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def ensure_data_dir() -> Path:
    """Ensure the database parent directory exists (idempotent)."""
    try:
        settings.db_data_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured data directory exists at {settings.db_data_dir}")
    except Exception:
        logger.exception(f"Failed to create data directory at {settings.db_data_dir}")
        raise
    return settings.db_data_dir


@lru_cache
def get_engine() -> Engine:
    """Lazily create SQLAlchemy engine after ensuring directory."""
    ensure_data_dir()
    logger.debug(f"Creating engine using SQLite DB: {settings.database_url}")
    return create_engine(url=settings.database_url, connect_args={"check_same_thread": False})


@event.listens_for(Engine, "connect")
def _set_sqlite_fk(dbapi_connection: Connection, _: Any) -> None:
    """Enable foreign key support for SQLite."""
    # https://docs.sqlalchemy.org/en/20/dialects/sqlite.html#foreign-key-support
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    """Session factory (cached)."""
    return sessionmaker(autocommit=False, autoflush=False, bind=get_engine())


def get_session() -> Generator[Session, Any]:
    """Dependency that yields a DB session."""
    factory = get_session_factory()
    with factory() as session:
        yield session


def run_db_migrations() -> None:
    """Run database migrations using Alembic."""
    ensure_data_dir()
    try:
        logger.info("Running database migrations...")
        alembic_cfg = Config(settings.alembic_config_path)
        alembic_cfg.set_main_option("script_location", settings.alembic_script_location)
        alembic_cfg.set_main_option("sqlalchemy.url", settings.database_url)
        command.upgrade(alembic_cfg, "head")
        logger.info("✓ Database migrations completed successfully")
    except Exception:
        logger.exception("✗ Database migration failed")
        raise
