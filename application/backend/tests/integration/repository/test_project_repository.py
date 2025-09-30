# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError

from db.models import ProjectDB
from repositories.project import ProjectRepository


@pytest.fixture
def repo(fxt_session):
    return ProjectRepository(session=fxt_session)


@pytest.fixture
def clean_after(request, fxt_clean_table):
    request.addfinalizer(lambda: fxt_clean_table(ProjectDB))


def test_add_and_get_by_id(repo, fxt_session, clean_after):
    p = ProjectDB(name="alpha", active=False)
    repo.add(p)
    fxt_session.commit()

    fetched = repo.get_by_id(p.id)
    assert fetched is not None
    assert fetched.id == p.id
    assert fetched.name == "alpha"
    assert fetched.active is False


def test_get_by_id_not_found(repo, clean_after):
    assert repo.get_by_id(uuid4()) is None


def test_get_all(repo, fxt_session, clean_after):
    names = {"p1", "p2", "p3"}
    for n in names:
        repo.add(ProjectDB(name=n))
    fxt_session.commit()

    all_projects = repo.get_all()
    assert {p.name for p in all_projects} == names


def test_exists_by_name_and_id(repo, fxt_session, clean_after):
    pid = uuid4()
    p = ProjectDB(id=pid, name="unique", active=False)
    repo.add(p)
    fxt_session.commit()

    assert repo.exists_by_name("unique") is True
    assert repo.exists_by_id(pid) is True


def test_exists_by_name_and_id_false(repo, fxt_session, clean_after):
    p = ProjectDB(id=uuid4(), name="another")
    repo.add(p)
    fxt_session.commit()

    assert repo.exists_by_name("absent") is False
    assert repo.exists_by_id(uuid4()) is False


def test_get_active_single(repo, fxt_session, clean_after):
    inactive = ProjectDB(name="inactive", active=False)
    active = ProjectDB(name="active", active=True)
    repo.add(inactive)
    repo.add(active)
    fxt_session.commit()

    result = repo.get_active()
    assert result is not None
    assert result.active is True
    assert result.name == "active"


def test_get_active_none(repo, fxt_session, clean_after):
    inactive = ProjectDB(name="inactive", active=False)
    repo.add(inactive)
    fxt_session.commit()
    assert repo.get_active() is None


def test_delete_project(repo, fxt_session, clean_after):
    p = ProjectDB(name="todelete", active=False)
    repo.add(p)
    fxt_session.commit()

    to_remove = repo.get_by_id(p.id)
    assert to_remove is not None
    repo.delete(to_remove)
    fxt_session.commit()

    assert repo.get_by_id(p.id) is None


def test_single_active_project_constraint(repo, fxt_session, clean_after):
    first = ProjectDB(name="active_primary", active=True)
    repo.add(first)
    fxt_session.commit()

    second = ProjectDB(name="active_secondary", active=True)
    repo.add(second)

    with pytest.raises(IntegrityError):
        fxt_session.commit()

    fxt_session.rollback()
    active_rows = fxt_session.query(ProjectDB).filter_by(active=True).all()
    assert len(active_rows) == 1
    assert active_rows[0].name == "active_primary"
