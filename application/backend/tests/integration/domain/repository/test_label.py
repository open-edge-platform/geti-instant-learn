# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

import pytest

from domain.db.models import AnnotationDB, LabelDB, ProjectDB, PromptDB, PromptType
from domain.repositories.label import LabelRepository


@pytest.fixture
def label_repository(fxt_session):
    return LabelRepository(session=fxt_session)


@pytest.fixture
def clean_after(fxt_clean_table):
    yield
    fxt_clean_table(AnnotationDB)
    fxt_clean_table(PromptDB)
    fxt_clean_table(LabelDB)
    fxt_clean_table(ProjectDB)


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


def test_get_label_ids_by_project_and_prompt_type_empty(label_repository, project_id, clean_after):
    result = label_repository.get_label_ids_by_project_and_prompt_type(project_id)
    assert len(result) == 0


def test_get_label_ids_by_project_and_prompt_type_visual_prompts(label_repository, fxt_session, clean_after):
    # Create project with unique name
    project = ProjectDB(name=f"test_project_{uuid4().hex[:8]}", active=False)
    fxt_session.add(project)
    fxt_session.commit()

    # Create labels
    label1 = LabelDB(id=uuid4(), name="car", project_id=project.id, color="#FF0000")
    label2 = LabelDB(id=uuid4(), name="person", project_id=project.id, color="#00FF00")
    label3 = LabelDB(id=uuid4(), name="unused", project_id=project.id, color="#0000FF")
    fxt_session.add_all([label1, label2, label3])
    fxt_session.commit()

    # Create visual prompts
    prompt1 = PromptDB(type=PromptType.VISUAL, project_id=project.id, frame_id=uuid4())
    prompt2 = PromptDB(type=PromptType.VISUAL, project_id=project.id, frame_id=uuid4())
    fxt_session.add_all([prompt1, prompt2])
    fxt_session.commit()

    # Create annotations linking prompts to labels
    ann1 = AnnotationDB(prompt_id=prompt1.id, label_id=label1.id, config={"type": "box"})
    ann2 = AnnotationDB(prompt_id=prompt1.id, label_id=label2.id, config={"type": "box"})
    ann3 = AnnotationDB(prompt_id=prompt2.id, label_id=label1.id, config={"type": "box"})
    fxt_session.add_all([ann1, ann2, ann3])
    fxt_session.commit()

    result = label_repository.get_label_ids_by_project_and_prompt_type(project.id, PromptType.VISUAL)

    assert len(result) == 2
    assert label1.id in result
    assert label2.id in result
    assert label3.id not in result  # unused label should not be included


def test_get_label_ids_by_project_and_prompt_type_filters_by_type(label_repository, fxt_session, clean_after):
    # Create project with unique name
    project = ProjectDB(name=f"test_project_{uuid4().hex[:8]}", active=False)
    fxt_session.add(project)
    fxt_session.commit()

    # Create labels
    label1 = LabelDB(id=uuid4(), name="car", project_id=project.id, color="#FF0000")
    label2 = LabelDB(id=uuid4(), name="person", project_id=project.id, color="#00FF00")
    fxt_session.add_all([label1, label2])
    fxt_session.commit()

    # Create prompts of different types
    visual_prompt = PromptDB(type=PromptType.VISUAL, project_id=project.id, frame_id=uuid4())
    text_prompt = PromptDB(type=PromptType.TEXT, project_id=project.id, text="find cars")
    fxt_session.add_all([visual_prompt, text_prompt])
    fxt_session.commit()

    # Create annotations
    ann1 = AnnotationDB(prompt_id=visual_prompt.id, label_id=label1.id, config={"type": "box"})
    ann2 = AnnotationDB(prompt_id=text_prompt.id, label_id=label2.id, config={"type": "box"})
    fxt_session.add_all([ann1, ann2])
    fxt_session.commit()

    # Query only visual prompts
    result = label_repository.get_label_ids_by_project_and_prompt_type(project.id, PromptType.VISUAL)
    assert len(result) == 1
    assert label1.id in result
    assert label2.id not in result

    # Query only text prompts
    result = label_repository.get_label_ids_by_project_and_prompt_type(project.id, PromptType.TEXT)
    assert len(result) == 1
    assert label2.id in result
    assert label1.id not in result

    # Query all prompt types
    result = label_repository.get_label_ids_by_project_and_prompt_type(project.id, None)
    assert len(result) == 2
    assert label1.id in result
    assert label2.id in result


def test_get_label_ids_by_project_and_prompt_type_filters_by_project(label_repository, fxt_session, clean_after):
    # Create projects with unique names
    project1 = ProjectDB(name=f"test_project_1_{uuid4().hex[:8]}", active=False)
    project2 = ProjectDB(name=f"test_project_2_{uuid4().hex[:8]}", active=False)
    fxt_session.add_all([project1, project2])
    fxt_session.commit()

    # Create labels for both projects
    label1 = LabelDB(id=uuid4(), name="car", project_id=project1.id, color="#FF0000")
    label2 = LabelDB(id=uuid4(), name="person", project_id=project2.id, color="#00FF00")
    fxt_session.add_all([label1, label2])
    fxt_session.commit()

    # Create prompts for both projects
    prompt1 = PromptDB(type=PromptType.VISUAL, project_id=project1.id, frame_id=uuid4())
    prompt2 = PromptDB(type=PromptType.VISUAL, project_id=project2.id, frame_id=uuid4())
    fxt_session.add_all([prompt1, prompt2])
    fxt_session.commit()

    # Create annotations
    ann1 = AnnotationDB(prompt_id=prompt1.id, label_id=label1.id, config={"type": "box"})
    ann2 = AnnotationDB(prompt_id=prompt2.id, label_id=label2.id, config={"type": "box"})
    fxt_session.add_all([ann1, ann2])
    fxt_session.commit()

    # Query project1
    result = label_repository.get_label_ids_by_project_and_prompt_type(project1.id, PromptType.VISUAL)
    assert len(result) == 1
    assert label1.id in result
    assert label2.id not in result
