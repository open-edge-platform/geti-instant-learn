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

logger = logging.getLogger(__name__)

DATABASE_URL = "sqlite:///./geti_prompt.db"  # SQLite file-based DB in project directory


logger.debug(f"Creating engine using SQLite DB: {DATABASE_URL}")
engine = create_engine(url=DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_session() -> Generator[Session, Any, None]:
    """Creates and returns database connection session"""
    with SessionLocal() as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]


def run_db_migrations() -> None:
    """Run database migrations using Alembic."""
    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")
