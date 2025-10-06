# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

import pytest

from core.components.schemas.reader import SourceType
from db.models import ProjectDB, SourceDB
from repositories.source import SourceRepository


@pytest.fixture
def repo(fxt_session):
    return SourceRepository(session=fxt_session)


@pytest.fixture
def clean_after(request, fxt_clean_table):
    # ensure both tables are cleaned (sources depend on projects)
    request.addfinalizer(lambda: fxt_clean_table(SourceDB))
    request.addfinalizer(lambda: fxt_clean_table(ProjectDB))


def _any_source_type():
    # use first enum member to stay resilient to enum definition changes
    return list(SourceType)[0]


def test_add_and_get_by_id(repo, fxt_session, clean_after):
    project = ProjectDB(name="proj")
    fxt_session.add(project)
    fxt_session.commit()

    src = SourceDB(name="camera-1", type=_any_source_type(), config={"url": "rtsp://x"}, project_id=project.id)
    repo.add(src)
    fxt_session.commit()

    fetched = repo.get_by_id(src.id)
    assert fetched is not None
    assert fetched.id == src.id
    assert fetched.project_id == project.id
    assert fetched.name == "camera-1"


def test_get_by_id_not_found(repo, fxt_session, clean_after):
    project = ProjectDB(name="proj")
    fxt_session.add(project)
    fxt_session.commit()

    src = SourceDB(name="camera-1", type=_any_source_type(), config={"url": "rtsp://x"}, project_id=project.id)
    repo.add(src)
    fxt_session.commit()

    assert repo.get_by_id(uuid4()) is None


def test_get_by_id_and_project(repo, fxt_session, clean_after):
    p1 = ProjectDB(name="p1")
    p2 = ProjectDB(name="p2")
    fxt_session.add_all([p1, p2])
    fxt_session.commit()

    src_p1 = SourceDB(name="s1", type=_any_source_type(), config={"a": 1}, project_id=p1.id)
    src_p2 = SourceDB(name="s2", type=_any_source_type(), config={"a": 2}, project_id=p2.id)
    repo.add(src_p1)
    repo.add(src_p2)
    fxt_session.commit()

    assert repo.get_by_id_and_project(src_p1.id, p1.id) is not None
    assert repo.get_by_id_and_project(src_p1.id, p2.id) is None  # wrong project scope


def test_get_all_by_project(repo, fxt_session, clean_after):
    p1 = ProjectDB(name="p1")
    p2 = ProjectDB(name="p2")
    fxt_session.add_all([p1, p2])
    fxt_session.commit()

    sources_names_p1 = {"s1", "s2", "s3"}
    for n in sources_names_p1:
        repo.add(SourceDB(name=n, type=_any_source_type(), config={"n": n}, project_id=p1.id))
    # extra in other project
    repo.add(SourceDB(name="other-source", type=_any_source_type(), config={}, project_id=p2.id))
    fxt_session.commit()

    result = repo.get_all_by_project(p1.id)
    assert {s.name for s in result} == sources_names_p1
    assert all(s.project_id == p1.id for s in result)


def test_delete_source(repo, fxt_session, clean_after):
    p = ProjectDB(name="del-proj")
    fxt_session.add(p)
    fxt_session.commit()

    s = SourceDB(name="todel", type=_any_source_type(), config={}, project_id=p.id)
    repo.add(s)
    fxt_session.commit()

    fetched = repo.get_by_id(s.id)
    assert fetched is not None

    repo.delete(fetched)
    fxt_session.commit()

    assert repo.get_by_id(s.id) is None


def test_get_connected_in_project(repo, fxt_session, clean_after):
    p = ProjectDB(name="proj")
    fxt_session.add(p)
    fxt_session.commit()

    repo.add(SourceDB(name="inactive", type=_any_source_type(), config={}, project_id=p.id, connected=False))
    repo.add(SourceDB(name="active", type=_any_source_type(), config={}, project_id=p.id, connected=True))
    fxt_session.commit()

    connected = repo.get_connected_in_project(p.id)
    assert connected is not None
    assert connected.connected is True
    assert connected.name == "active"


def test_project_deletion_cascades_sources(repo, fxt_session, clean_after):
    """
    Ensure relationship cascade (all, delete-orphan) removes sources when their project is deleted.
    """
    project = ProjectDB(name="proj")
    fxt_session.add(project)
    fxt_session.commit()

    src_ids = []
    for i in range(3):
        s = SourceDB(name=f"s{i}", type=_any_source_type(), config={"i": i}, project_id=project.id)
        repo.add(s)
        src_ids.append(s.id)
    fxt_session.commit()
    assert len(repo.get_all_by_project(project.id)) == 3

    # delete project -> should cascade to sources
    fxt_session.delete(project)
    fxt_session.commit()

    # all sources should be gone
    for sid in src_ids:
        assert repo.get_by_id(sid) is None
    # get_all_by_project should return an empty list
    assert repo.get_all_by_project(project.id) == []
