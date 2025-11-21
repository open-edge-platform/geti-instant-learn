# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError

from domain.db.models import ProjectDB, SourceDB
from domain.repositories.source import SourceRepository
from domain.services.schemas.reader import SourceType


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


def test_get_connected_in_project_none(source_repo, fxt_session, clean_after):
    project = ProjectDB(name="proj")
    fxt_session.add(project)
    fxt_session.commit()

    source_repo.add(make_source(project.id, label="disconnected"))
    fxt_session.commit()

    connected = source_repo.get_connected_in_project(project.id)
    assert connected is None


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


def test_single_connected_source_per_project(source_repo, fxt_session, clean_after):
    project = ProjectDB(name="proj")
    fxt_session.add(project)
    fxt_session.commit()

    types = list(SourceType)
    first_type = types[0]
    second_type = types[1] if len(types) > 1 else types[0]

    first = SourceDB(
        config={"source_type": first_type, "label": "first"},
        project_id=project.id,
        connected=True,
    )
    source_repo.add(first)
    fxt_session.commit()

    # try to create second connected source (should fail if same type provided, so use different type)
    second = SourceDB(
        config={"source_type": second_type, "label": "second"},
        project_id=project.id,
        connected=True,
    )
    source_repo.add(second)

    with pytest.raises(IntegrityError):
        fxt_session.commit()

    fxt_session.rollback()

    connected = source_repo.get_connected_in_project(project.id)
    assert connected is not None
    assert connected.config.get("label") == "first"


def test_multiple_disconnected_sources_allowed(source_repo, fxt_session, clean_after):
    project = ProjectDB(name="proj")
    fxt_session.add(project)
    fxt_session.commit()

    types = list(SourceType)
    for i, st in enumerate(types[:3]):
        src = SourceDB(
            config={"source_type": st, "label": f"source_{i}"},
            project_id=project.id,
            connected=False,
        )
        source_repo.add(src)
    fxt_session.commit()

    all_sources = source_repo.get_all_by_project(project.id)
    assert len(all_sources) == 3
    assert all(not s.connected for s in all_sources)


def test_unique_source_name_per_project(source_repo, fxt_session, clean_after):
    project = ProjectDB(name="proj")
    fxt_session.add(project)
    fxt_session.commit()

    types = list(SourceType)
    first_type = types[0]
    second_type = types[1] if len(types) > 1 else types[0]

    first = SourceDB(
        config={"source_type": first_type, "name": "my_source"},
        project_id=project.id,
    )
    source_repo.add(first)
    fxt_session.commit()

    # try to create second source with same name (different type to avoid type constraint)
    second = SourceDB(
        config={"source_type": second_type, "name": "my_source"},
        project_id=project.id,
    )
    source_repo.add(second)

    with pytest.raises(IntegrityError):
        fxt_session.commit()

    fxt_session.rollback()


def test_source_name_optional(source_repo, fxt_session, clean_after):
    project = ProjectDB(name="proj")
    fxt_session.add(project)
    fxt_session.commit()

    types = list(SourceType)
    for st in types[:2]:
        src = SourceDB(
            config={"source_type": st},
            project_id=project.id,
        )
        source_repo.add(src)
    fxt_session.commit()

    all_sources = source_repo.get_all_by_project(project.id)
    assert len(all_sources) == 2


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
