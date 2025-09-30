# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from alembic import command
from alembic.config import Config
from settings import get_settings


def _run_migrations(db_url: str) -> None:
    settings = get_settings()
    alembic_cfg = Config(str(settings.alembic_config_path))
    alembic_cfg.set_main_option("script_location", str(settings.alembic_script_location))
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)
    command.upgrade(alembic_cfg, "head")


@pytest.fixture(scope="session")
def fxt_db_url(tmp_path_factory) -> str:
    db_dir = tmp_path_factory.mktemp("db")
    db_file = db_dir / "test_geti_prompt.sqlite"
    return f"sqlite:///{db_file}"


@pytest.fixture(scope="session", autouse=True)
def fxt_migrated_db(fxt_db_url: str):
    _run_migrations(fxt_db_url)
    yield


@pytest.fixture(scope="session")
def fxt_engine(fxt_db_url: str):
    engine = create_engine(fxt_db_url, connect_args={"check_same_thread": False})
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def fxt_session_maker(fxt_engine):
    return sessionmaker(autocommit=False, autoflush=False, bind=fxt_engine)


@pytest.fixture
def fxt_session(fxt_session_maker):
    session = fxt_session_maker()
    try:
        yield session
        session.rollback()
    finally:
        session.close()


@pytest.fixture
def fxt_clean_table(fxt_session):
    def _clean(model_cls):
        fxt_session.query(model_cls).delete()
        fxt_session.commit()

    return _clean
