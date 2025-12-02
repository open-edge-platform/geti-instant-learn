# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError

from domain.db.models import ProcessorDB, ProjectDB
from domain.repositories.processor import ProcessorRepository


@pytest.fixture
def processor_repo(fxt_session):
    return ProcessorRepository(session=fxt_session)


@pytest.fixture
def clean_after(request, fxt_clean_table):
    request.addfinalizer(lambda: fxt_clean_table(ProcessorDB))
    request.addfinalizer(lambda: fxt_clean_table(ProjectDB))


def make_processor(project_id, name=None, active=False, **extra_cfg) -> ProcessorDB:
    cfg = {"type": "sam2", **extra_cfg}
    return ProcessorDB(name=name, config=cfg, project_id=project_id, active=active)


def test_add_and_get_by_id(processor_repo, fxt_session, clean_after):
    project = ProjectDB(name="proj")
    fxt_session.add(project)
    fxt_session.commit()

    proc = make_processor(project.id, name="proc1")
    processor_repo.add(proc)
    fxt_session.commit()

    fetched = processor_repo.get_by_id(proc.id)
    assert fetched is not None
    assert fetched.id == proc.id
    assert fetched.name == "proc1"


def test_get_by_id_not_found(processor_repo, clean_after):
    assert processor_repo.get_by_id(uuid4()) is None


def test_get_by_id_and_project(processor_repo, fxt_session, clean_after):
    project_a = ProjectDB(name="A")
    project_b = ProjectDB(name="B")
    fxt_session.add_all([project_a, project_b])
    fxt_session.commit()

    proc_a = make_processor(project_a.id)
    proc_b = make_processor(project_b.id)
    processor_repo.add(proc_a)
    processor_repo.add(proc_b)
    fxt_session.commit()

    assert processor_repo.get_by_id_and_project(proc_a.id, project_a.id) is not None
    assert processor_repo.get_by_id_and_project(proc_a.id, project_b.id) is None


def test_list_all_by_project(processor_repo, fxt_session, clean_after):
    project_main = ProjectDB(name="main")
    project_other = ProjectDB(name="other")
    fxt_session.add_all([project_main, project_other])
    fxt_session.commit()

    procs = [make_processor(project_main.id, name=f"proc{i}") for i in range(3)]
    for p in procs:
        processor_repo.add(p)
    processor_repo.add(make_processor(project_other.id))
    fxt_session.commit()

    result = processor_repo.list_all_by_project(project_main.id)
    assert len(result) == 3
    assert {p.id for p in result} == {p.id for p in procs}


def test_delete(processor_repo, fxt_session, clean_after):
    project = ProjectDB(name="proj")
    fxt_session.add(project)
    fxt_session.commit()

    proc = make_processor(project.id)
    processor_repo.add(proc)
    fxt_session.commit()

    assert processor_repo.get_by_id(proc.id) is not None
    deleted = processor_repo.delete(proc.id)
    fxt_session.commit()

    assert deleted is True
    assert processor_repo.get_by_id(proc.id) is None


def test_get_active_in_project(processor_repo, fxt_session, clean_after):
    project = ProjectDB(name="proj", active=True)
    fxt_session.add(project)
    fxt_session.commit()

    processor_repo.add(make_processor(project.id, name="inactive"))
    active = make_processor(project.id, name="active", active=True)
    processor_repo.add(active)
    fxt_session.commit()

    result = processor_repo.get_active_in_project(project.id)
    assert result is not None
    assert result.active is True
    assert result.name == "active"


def test_get_active_in_project_none(processor_repo, fxt_session, clean_after):
    project = ProjectDB(name="proj", active=True)
    fxt_session.add(project)
    fxt_session.commit()

    processor_repo.add(make_processor(project.id, name="inactive"))
    fxt_session.commit()

    result = processor_repo.get_active_in_project(project.id)
    assert result is None


def test_project_deletion_cascades_processors(processor_repo, fxt_session, clean_after):
    project = ProjectDB(name="proj")
    fxt_session.add(project)
    fxt_session.commit()

    procs = [make_processor(project.id) for _ in range(3)]
    for p in procs:
        processor_repo.add(p)
    fxt_session.commit()

    proc_ids = [p.id for p in procs]

    fxt_session.delete(project)
    fxt_session.commit()

    for pid in proc_ids:
        assert processor_repo.get_by_id(pid) is None


def test_single_active_processor_per_project(processor_repo, fxt_session, clean_after):
    project = ProjectDB(name="proj", active=True)
    fxt_session.add(project)
    fxt_session.commit()

    first = make_processor(project.id, name="first", active=True)
    processor_repo.add(first)
    fxt_session.commit()

    second = make_processor(project.id, name="second", active=True)
    with pytest.raises(IntegrityError):
        processor_repo.add(second)
        fxt_session.flush()
    fxt_session.rollback()

    result = processor_repo.get_active_in_project(project.id)
    assert result is not None
    assert result.name == "first"


def test_multiple_inactive_processors_allowed(processor_repo, fxt_session, clean_after):
    project = ProjectDB(name="proj")
    fxt_session.add(project)
    fxt_session.commit()

    for i in range(3):
        proc = make_processor(project.id, name=f"proc{i}", active=False)
        processor_repo.add(proc)
    fxt_session.commit()

    all_procs = processor_repo.list_all_by_project(project.id)
    assert len(all_procs) == 3
    assert all(not p.active for p in all_procs)


def test_unique_processor_name_per_project(processor_repo, fxt_session, clean_after):
    project = ProjectDB(name="proj")
    fxt_session.add(project)
    fxt_session.commit()

    first = make_processor(project.id, name="my_processor")
    processor_repo.add(first)
    fxt_session.commit()

    second = make_processor(project.id, name="my_processor")
    with pytest.raises(IntegrityError):
        processor_repo.add(second)
        fxt_session.flush()
    fxt_session.rollback()


def test_processor_name_optional(processor_repo, fxt_session, clean_after):
    project = ProjectDB(name="proj")
    fxt_session.add(project)
    fxt_session.commit()

    for i in range(2):
        proc = make_processor(project.id, name=None)
        processor_repo.add(proc)
    fxt_session.commit()

    all_procs = processor_repo.list_all_by_project(project.id)
    assert len(all_procs) == 2


def test_active_default_false(processor_repo, fxt_session, clean_after):
    project = ProjectDB(name="proj")
    fxt_session.add(project)
    fxt_session.commit()

    proc = make_processor(project.id)
    processor_repo.add(proc)
    fxt_session.commit()

    fetched = processor_repo.get_by_id(proc.id)
    assert fetched is not None
    assert fetched.active is False
