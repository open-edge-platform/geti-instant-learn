# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Generator
from typing import Annotated, Any

from fastapi import Depends
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from alembic import command
from alembic.config import Config
from db.models import ProjectDB
from services.common import ResourceNotFoundError
from services.project import ProjectService
from settings import Settings

logger = logging.getLogger(__name__)

DEFAULT_PROJECT_NAME = "Project #1"
DATABASE_URL = "sqlite:///./geti_prompt.db"  # SQLite file-based DB in project directory


logger.debug(f"Creating engine using SQLite DB: {DATABASE_URL}")
engine = create_engine(url=DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_session() -> Generator[Session, Any]:
    """Creates and returns database connection session"""
    with SessionLocal() as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]


def run_db_migrations(settings: Settings) -> None:
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


def ensure_default_active_project() -> None:
    """
    Ensure there is exactly one active project.
    Create or activate the default one if missing.
    """
    with SessionLocal() as session:
        service = ProjectService(session)
        try:
            service.get_active_project()
            # if an active project exists, nothing to do
            return
        except ResourceNotFoundError:
            pass  # proceed to create / activate

        try:
            existing = session.query(ProjectDB).filter_by(name=DEFAULT_PROJECT_NAME).one_or_none()
            if existing:
                logger.info(f"Activating existing default project '{DEFAULT_PROJECT_NAME}'")
                service.set_active_project(existing.id)
            else:
                logger.info(f"Creating and activating default project '{DEFAULT_PROJECT_NAME}'")
                service.create_project(ProjectDB(name=DEFAULT_PROJECT_NAME))
        except Exception:
            logger.exception("Failed to create default active project at the application startup")
            session.rollback()
