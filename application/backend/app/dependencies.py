# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Generator
from functools import lru_cache
from pathlib import Path
from sqlite3 import Connection
from typing import Annotated, Any

from fastapi import Depends, Request
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from alembic import command
from alembic.config import Config
from core.runtime.dispatcher import ConfigChangeDispatcher
from core.runtime.pipeline_manager import PipelineManager
from repositories.frame import FrameRepository
from repositories.project import ProjectRepository
from repositories.source import SourceRepository
from services.frame import FrameService
from settings import get_settings
from webrtc.manager import WebRTCManager

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


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    """Session factory (cached)."""
    return sessionmaker(autocommit=False, autoflush=False, bind=get_engine())


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection: Connection, _: Any) -> None:
    """Enable foreign key support for SQLite."""
    # https://docs.sqlalchemy.org/en/20/dialects/sqlite.html#foreign-key-support
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def get_session() -> Generator[Session, Any]:
    """Dependency that yields a DB session."""
    SessionLocal = get_session_factory()
    with SessionLocal() as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]


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


def get_pipeline_manager(request: Request) -> PipelineManager:
    """Dependency that provides access to the PipelineManager."""
    return request.app.state.pipeline_manager


def get_webrtc_manager(request: Request) -> WebRTCManager:
    """Provides the global WebRTCManager instance from FastAPI application's state."""
    return request.app.state.webrtc_manager


def get_config_dispatcher(request: Request) -> ConfigChangeDispatcher:
    """Dependency that provides access to the ConfigChangeDispatcher."""
    return request.app.state.config_dispatcher


ConfigChangeDispatcherDep = Annotated[ConfigChangeDispatcher, Depends(get_config_dispatcher)]


def get_frame_repository() -> FrameRepository:
    """Dependency that provides a FrameRepository instance."""
    return FrameRepository()


def get_project_repository(session: SessionDep) -> ProjectRepository:
    """Dependency that provides a ProjectRepository instance."""
    return ProjectRepository(session)


def get_source_repository(session: SessionDep) -> SourceRepository:
    """Dependency that provides a SourceRepository instance."""
    return SourceRepository(session)


def get_frame_service(
    pipeline_manager: Annotated[PipelineManager, Depends(get_pipeline_manager)],
    frame_repo: Annotated[FrameRepository, Depends(get_frame_repository)],
    project_repo: Annotated[ProjectRepository, Depends(get_project_repository)],
    source_repo: Annotated[SourceRepository, Depends(get_source_repository)],
) -> FrameService:
    """Dependency that provides a FrameService instance."""
    return FrameService(pipeline_manager, frame_repo, project_repo, source_repo)
