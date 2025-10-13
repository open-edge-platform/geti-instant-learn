# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Generator
from sqlite3 import Connection
from typing import Annotated, Any

from fastapi import Depends
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from alembic import command
from alembic.config import Config
from settings import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

logger.debug(f"Creating engine using SQLite DB: {settings.database_url}")
engine = create_engine(url=settings.database_url, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection: Connection, _: Any) -> None:
    """Enable foreign key support for SQLite."""
    # https://docs.sqlalchemy.org/en/20/dialects/sqlite.html#foreign-key-support
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def get_session() -> Generator[Session, Any]:
    """Creates and returns database connection session"""
    with SessionLocal() as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]


def run_db_migrations() -> None:
    """Run database migrations using Alembic."""
    try:
        logger.info("Running database migrations...")
        alembic_cfg = Config(settings.alembic_config_path)
        alembic_cfg.set_main_option("script_location", settings.alembic_script_location)
        alembic_cfg.set_main_option("sqlalchemy.url", settings.database_url)
        command.upgrade(alembic_cfg, "head")
        logger.info("✓ Database migrations completed successfully")
    except Exception:
        logger.exception("✗ Database migration failed")
