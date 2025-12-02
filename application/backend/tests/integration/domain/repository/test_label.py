# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

import pytest

from domain.db.models import LabelDB, ProjectDB
from domain.repositories.label import LabelRepository


@pytest.fixture
def label_repository(fxt_session):
    return LabelRepository(session=fxt_session)


@pytest.fixture
def clean_after(request, fxt_clean_table):
    request.addfinalizer(lambda: fxt_clean_table(LabelDB))
    request.addfinalizer(lambda: fxt_clean_table(ProjectDB))


@pytest.fixture
def projects(fxt_session):
    project = ProjectDB(name="test_project", active=False)
    other_project = ProjectDB(name="other_project", active=False)
    fxt_session.add_all([project, other_project])
    fxt_session.commit()
    return project, other_project


@pytest.fixture
def project_id(projects):
    return projects[0].id


@pytest.fixture
def sample_label(project_id):
    return LabelDB(id=uuid4(), name="test_label", project_id=project_id, color="#FF0000")


def test_add_label(label_repository, sample_label, fxt_session, clean_after):
    label_repository.add(sample_label)
    fxt_session.commit()

    result = label_repository.get_by_id_and_project(sample_label.id, sample_label.project_id)
    assert result is not None
    assert result.name == "test_label"
    assert result.id == sample_label.id


def test_get_by_id_and_project_exists(label_repository, sample_label, fxt_session, clean_after):
    fxt_session.add(sample_label)
    fxt_session.commit()

    result = label_repository.get_by_id_and_project(sample_label.id, sample_label.project_id)
    assert result is not None
    assert result.id == sample_label.id
    assert result.name == "test_label"


def test_get_by_id_and_project_not_exists(label_repository, project_id, clean_after):
    result = label_repository.get_by_id_and_project(uuid4(), project_id)
    assert result is None


def test_get_by_id_and_project_wrong_project(label_repository, sample_label, fxt_session, clean_after):
    label_repository.add(sample_label)
    fxt_session.commit()

    result = label_repository.get_by_id_and_project(sample_label.id, uuid4())
    assert result is None


def test_list_all_by_project_empty(label_repository, project_id, clean_after):
    result = label_repository.list_all_by_project(project_id)
    assert len(result) == 0


def test_list_all_by_project_multiple(label_repository, project_id, fxt_session, clean_after):
    labels = [LabelDB(id=uuid4(), name=f"label_{i}", project_id=project_id, color="#000000") for i in range(3)]
    for label in labels:
        label_repository.add(label)
    fxt_session.commit()

    result = label_repository.list_all_by_project(project_id)
    assert len(result) == 3


def test_list_all_by_project_filters_by_project(label_repository, projects, fxt_session, clean_after):
    project, other_project = projects

    label_repository.add(LabelDB(id=uuid4(), name="label_1", project_id=project.id, color="#000000"))
    label_repository.add(LabelDB(id=uuid4(), name="label_2", project_id=other_project.id, color="#000000"))
    fxt_session.commit()

    result = label_repository.list_all_by_project(project.id)
    assert len(result) == 1
    assert result[0].name == "label_1"


def test_delete_label(label_repository, sample_label, fxt_session, clean_after):
    label_repository.add(sample_label)
    fxt_session.commit()

    deleted = label_repository.delete(sample_label.id)
    fxt_session.commit()

    assert deleted is True
    assert label_repository.get_by_id(sample_label.id) is None


def test_delete_not_found(label_repository, clean_after):
    deleted = label_repository.delete(uuid4())
    assert deleted is False


def test_list_with_pagination_by_project_empty(label_repository, project_id, clean_after):
    labels, total_count = label_repository.list_with_pagination_by_project(project_id)
    assert len(labels) == 0
    assert total_count == 0


def test_list_with_pagination_by_project_first_page(label_repository, project_id, fxt_session, clean_after):
    for i in range(25):
        label_repository.add(LabelDB(id=uuid4(), name=f"label_{i:02d}", project_id=project_id, color="#000000"))
    fxt_session.commit()

    labels, total_count = label_repository.list_with_pagination_by_project(project_id, offset=0, limit=20)
    assert len(labels) == 20
    assert total_count == 25


def test_list_with_pagination_by_project_second_page(label_repository, project_id, fxt_session, clean_after):
    for i in range(25):
        label_repository.add(LabelDB(id=uuid4(), name=f"label_{i:02d}", project_id=project_id, color="#000000"))
    fxt_session.commit()

    labels, total_count = label_repository.list_with_pagination_by_project(project_id, offset=20, limit=20)
    assert len(labels) == 5
    assert total_count == 25


def test_list_with_pagination_by_project_filters(label_repository, projects, fxt_session, clean_after):
    project, other_project = projects

    label_repository.add(LabelDB(id=uuid4(), name="label_1", project_id=project.id, color="#000000"))
    label_repository.add(LabelDB(id=uuid4(), name="label_2", project_id=other_project.id, color="#000000"))
    fxt_session.commit()

    labels, total_count = label_repository.list_with_pagination_by_project(project.id)
    assert len(labels) == 1
    assert total_count == 1
