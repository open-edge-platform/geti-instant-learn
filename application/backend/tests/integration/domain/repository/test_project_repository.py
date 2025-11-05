# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError

from domain.db.models import ProjectDB
from domain.repositories.project import ProjectRepository


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


def test_unique_project_name_constraint(repo, fxt_session, clean_after):
    first = ProjectDB(name="duplicate_name", active=False)
    repo.add(first)
    fxt_session.commit()

    second = ProjectDB(name="duplicate_name", active=False)
    repo.add(second)

    with pytest.raises(IntegrityError):
        fxt_session.commit()

    fxt_session.rollback()
    projects = fxt_session.query(ProjectDB).filter_by(name="duplicate_name").all()
    assert len(projects) == 1


def test_get_paginated_empty(repo, fxt_session, clean_after):
    projects, total = repo.get_paginated(offset=0, limit=10)
    assert len(projects) == 0
    assert total == 0


def test_get_paginated_first_page(repo, fxt_session, clean_after):
    for i in range(5):
        repo.add(ProjectDB(name=f"project_{i}"))
    fxt_session.commit()

    projects, total = repo.get_paginated(offset=0, limit=3)
    assert len(projects) == 3
    assert total == 5
    names = [p.name for p in projects]
    assert names == sorted(names)


def test_get_paginated_second_page(repo, fxt_session, clean_after):
    for i in range(5):
        repo.add(ProjectDB(name=f"project_{i}"))
    fxt_session.commit()

    projects, total = repo.get_paginated(offset=3, limit=3)
    assert len(projects) == 2
    assert total == 5


def test_get_paginated_beyond_end(repo, fxt_session, clean_after):
    for i in range(3):
        repo.add(ProjectDB(name=f"project_{i}"))
    fxt_session.commit()

    projects, total = repo.get_paginated(offset=10, limit=5)
    assert len(projects) == 0
    assert total == 3


def test_get_paginated_ordering(repo, fxt_session, clean_after):
    names = ["zebra", "alpha", "beta", "gamma"]
    for name in names:
        repo.add(ProjectDB(name=name))
    fxt_session.commit()

    projects, total = repo.get_paginated(offset=0, limit=10)
    result_names = [p.name for p in projects]
    assert result_names == ["alpha", "beta", "gamma", "zebra"]
    assert total == 4
