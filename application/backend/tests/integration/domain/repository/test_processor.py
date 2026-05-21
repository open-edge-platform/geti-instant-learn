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


def make_processor(project_id, name=None, active=False, prompt_mode="VISUAL", **extra_cfg) -> ProcessorDB:
    cfg = {"type": "sam2", **extra_cfg}
    return ProcessorDB(name=name, config=cfg, project_id=project_id, active=active, prompt_mode=prompt_mode)


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

    # Same name + same mode → constraint violation
    first = make_processor(project.id, name="my_processor", prompt_mode="VISUAL")
    processor_repo.add(first)
    fxt_session.commit()

    second = make_processor(project.id, name="my_processor", prompt_mode="VISUAL")
    with pytest.raises(IntegrityError):
        processor_repo.add(second)
        fxt_session.flush()
    fxt_session.rollback()


def test_same_name_different_mode_allowed(processor_repo, fxt_session, clean_after):
    """Same processor name with different prompt_mode is allowed (dual-mode models like SAM3)."""
    project = ProjectDB(name="proj")
    fxt_session.add(project)
    fxt_session.commit()

    visual = make_processor(project.id, name="sam3", prompt_mode="VISUAL")
    text = make_processor(project.id, name="sam3", prompt_mode="TEXT")
    processor_repo.add(visual)
    processor_repo.add(text)
    fxt_session.commit()

    result = processor_repo.list_all_by_project(project.id)
    assert len(result) == 2


def test_list_by_project_and_mode(processor_repo, fxt_session, clean_after):
    """list_by_project_and_mode returns only processors for the given mode, newest first."""
    project = ProjectDB(name="proj", active=True)
    fxt_session.add(project)
    fxt_session.commit()

    visual1 = make_processor(project.id, name="matcher", prompt_mode="VISUAL")
    visual2 = make_processor(project.id, name="soft_matcher", prompt_mode="VISUAL")
    text1 = make_processor(project.id, name="sam3", prompt_mode="TEXT")
    for p in [visual1, visual2, text1]:
        processor_repo.add(p)
    fxt_session.commit()

    visual_result = processor_repo.list_by_project_and_mode(project.id, "VISUAL")
    assert len(visual_result) == 2
    assert all(p.prompt_mode == "VISUAL" for p in visual_result)

    text_result = processor_repo.list_by_project_and_mode(project.id, "TEXT")
    assert len(text_result) == 1
    assert text_result[0].name == "sam3"


def test_list_with_pagination_by_project_and_mode(processor_repo, fxt_session, clean_after):
    """list_with_pagination_by_project_and_mode paginates correctly within a mode."""
    project = ProjectDB(name="proj", active=True)
    fxt_session.add(project)
    fxt_session.commit()

    for i in range(3):
        processor_repo.add(make_processor(project.id, name=f"visual{i}", prompt_mode="VISUAL"))
    processor_repo.add(make_processor(project.id, name="sam3", prompt_mode="TEXT"))
    fxt_session.commit()

    items, total = processor_repo.list_with_pagination_by_project_and_mode(project.id, "VISUAL", offset=0, limit=2)
    assert total == 3
    assert len(items) == 2

    items_all, total_all = processor_repo.list_with_pagination_by_project_and_mode(
        project.id, "VISUAL", offset=0, limit=10
    )
    assert total_all == 3
    assert len(items_all) == 3


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
