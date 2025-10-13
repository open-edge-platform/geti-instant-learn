# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError

from core.components.schemas.reader import SourceType
from db.models import ProjectDB, SourceDB
from repositories.source import SourceRepository


@pytest.fixture
def source_repo(fxt_session):
    return SourceRepository(session=fxt_session)


@pytest.fixture
def clean_after(request, fxt_clean_table):
    request.addfinalizer(lambda: fxt_clean_table(SourceDB))
    request.addfinalizer(lambda: fxt_clean_table(ProjectDB))


def any_source_type() -> SourceType:
    return list(SourceType)[0]


def make_source(project_id, source_type: SourceType | None = None, **extra_cfg) -> SourceDB:
    st = source_type or any_source_type()
    cfg = {"source_type": st, **extra_cfg}
    return SourceDB(config=cfg, project_id=project_id)


def test_add_and_get_by_id(source_repo, fxt_session, clean_after):
    project = ProjectDB(name="proj")
    fxt_session.add(project)
    fxt_session.commit()

    src = make_source(project.id, url="rtsp://cam/1")
    source_repo.add(src)
    fxt_session.commit()

    fetched = source_repo.get_by_id(src.id)
    assert fetched is not None
    assert fetched.id == src.id
    assert fetched.project_id == project.id
    assert fetched.config["url"] == "rtsp://cam/1"
    assert fetched.config["source_type"] == src.config["source_type"]


def test_get_by_id_not_found(source_repo, fxt_session, clean_after):
    project = ProjectDB(name="proj")
    fxt_session.add(project)
    fxt_session.commit()

    src = make_source(project.id)
    source_repo.add(src)
    fxt_session.commit()

    assert source_repo.get_by_id(uuid4()) is None


def test_get_by_id_and_project(source_repo, fxt_session, clean_after):
    project_a = ProjectDB(name="A")
    project_b = ProjectDB(name="B")
    fxt_session.add_all([project_a, project_b])
    fxt_session.commit()

    src_a = make_source(project_a.id)
    src_b = make_source(project_b.id)
    source_repo.add(src_a)
    source_repo.add(src_b)
    fxt_session.commit()

    assert source_repo.get_by_id_and_project(src_a.id, project_a.id) is not None
    assert source_repo.get_by_id_and_project(src_a.id, project_b.id) is None


def test_get_all_by_project(source_repo, fxt_session, clean_after):
    project_main = ProjectDB(name="main")
    project_other = ProjectDB(name="other")
    fxt_session.add_all([project_main, project_other])
    fxt_session.commit()

    # use distinct source types to satisfy uniqueness constraint.
    all_types = list(SourceType)
    added = []
    for st in all_types[:3]:
        s = make_source(project_main.id, source_type=st, idx=len(added))
        source_repo.add(s)
        added.append(s)

    source_repo.add(make_source(project_other.id))
    fxt_session.commit()

    result = source_repo.get_all_by_project(project_main.id)
    assert {s.id for s in result} == {s.id for s in added}
    assert all(s.project_id == project_main.id for s in result)


def test_delete_source(source_repo, fxt_session, clean_after):
    project = ProjectDB(name="del")
    fxt_session.add(project)
    fxt_session.commit()

    src = make_source(project.id)
    source_repo.add(src)
    fxt_session.commit()

    assert source_repo.get_by_id(src.id) is not None
    source_repo.delete(src)
    fxt_session.commit()
    assert source_repo.get_by_id(src.id) is None


def test_get_connected_in_project(source_repo, fxt_session, clean_after):
    project = ProjectDB(name="proj")
    fxt_session.add(project)
    fxt_session.commit()

    tlist = list(SourceType)
    inactive_type = tlist[0]
    active_type = tlist[1] if len(tlist) > 1 else tlist[0]

    source_repo.add(make_source(project.id, source_type=inactive_type, label="inactive"))
    active = SourceDB(
        config={"source_type": active_type, "label": "active"},
        project_id=project.id,
        connected=True,
    )
    source_repo.add(active)
    fxt_session.commit()

    connected = source_repo.get_connected_in_project(project.id)
    assert connected is not None
    assert connected.connected is True
    assert connected.config.get("label") == "active"


def test_get_by_type_in_project(source_repo, fxt_session, clean_after):
    project = ProjectDB(name="proj")
    fxt_session.add(project)
    fxt_session.commit()

    types = list(SourceType)
    primary_type = types[0]
    other_type = types[1] if len(types) > 1 else primary_type

    primary_src = make_source(project.id, source_type=primary_type, tag="primary")
    other_src = make_source(project.id, source_type=other_type, tag="other")
    source_repo.add(primary_src)
    source_repo.add(other_src)
    fxt_session.commit()

    fetched = source_repo.get_by_type_in_project(project.id, primary_type)
    assert fetched is not None
    assert fetched.config["source_type"] == primary_type
    assert fetched.config.get("tag") == "primary"


def test_project_deletion_cascades_sources(source_repo, fxt_session, clean_after):
    project = ProjectDB(name="proj")
    fxt_session.add(project)
    fxt_session.commit()

    created = []
    types = list(SourceType)
    for i, st in enumerate(types[:3]):
        s = make_source(project.id, source_type=st, idx=i)
        source_repo.add(s)
        created.append(s)
    fxt_session.commit()
    assert len(source_repo.get_all_by_project(project.id)) == len(created)

    created_ids = [s.id for s in created]

    fxt_session.delete(project)
    fxt_session.commit()

    for sid in created_ids:
        assert source_repo.get_by_id(sid) is None
    assert source_repo.get_all_by_project(project.id) == []


def test_unique_source_type_per_project(source_repo, fxt_session, clean_after):
    project = ProjectDB(name="unique")
    fxt_session.add(project)
    fxt_session.commit()

    st = any_source_type()
    first = make_source(project.id, source_type=st, label="first")
    second = make_source(project.id, source_type=st, label="second")

    source_repo.add(first)
    fxt_session.commit()
    source_repo.add(second)

    with pytest.raises(IntegrityError):
        fxt_session.commit()
    fxt_session.rollback()

    fetched = source_repo.get_by_type_in_project(project.id, st)
    assert fetched is not None
    assert fetched.config.get("label") == "first"


def test_connected_default_false(source_repo, fxt_session, clean_after):
    project = ProjectDB(name="defaults")
    fxt_session.add(project)
    fxt_session.commit()

    src = make_source(project.id)
    source_repo.add(src)
    fxt_session.commit()

    fetched = source_repo.get_by_id(src.id)
    assert fetched is not None
    assert fetched.connected is False
