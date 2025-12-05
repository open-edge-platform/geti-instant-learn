# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError

from domain.db.models import AnnotationDB, LabelDB, ProjectDB, PromptDB, PromptType
from domain.repositories.prompt import PromptRepository


@pytest.fixture
def prompt_repo(fxt_session):
    return PromptRepository(session=fxt_session)


@pytest.fixture
def clean_after(fxt_clean_table):
    """Cleanup fixture - runs after test completion."""
    yield
    fxt_clean_table(AnnotationDB)
    fxt_clean_table(PromptDB)
    fxt_clean_table(LabelDB)
    fxt_clean_table(ProjectDB)


def make_project(name=None) -> ProjectDB:
    """Create a project with a unique name."""
    return ProjectDB(name=name or f"proj-{uuid4().hex[:8]}")


def make_text_prompt(project_id, text="test prompt") -> PromptDB:
    return PromptDB(
        type=PromptType.TEXT,
        text=text,
        frame_id=None,
        project_id=project_id,
    )


def make_visual_prompt(project_id, frame_id=None, annotations=None) -> PromptDB:
    frame_id = frame_id or uuid4()
    prompt = PromptDB(
        type=PromptType.VISUAL,
        text=None,
        frame_id=frame_id,
        project_id=project_id,
    )
    if annotations:
        prompt.annotations = annotations
    return prompt


def make_annotation(prompt_id, label_id, config=None) -> AnnotationDB:
    if config is None:
        config = {"type": "rectangle", "points": [{"x": 0.1, "y": 0.1}, {"x": 0.5, "y": 0.5}]}
    return AnnotationDB(
        config=config,
        label_id=label_id,
        prompt_id=prompt_id,
    )


def test_add_and_get_by_id_text_prompt(prompt_repo, fxt_session, clean_after):
    project = make_project()
    fxt_session.add(project)
    fxt_session.commit()

    prompt = make_text_prompt(project.id, text="find red car")
    prompt_repo.add(prompt)
    fxt_session.commit()

    fetched = prompt_repo.get_by_id(prompt.id)
    assert fetched is not None
    assert fetched.id == prompt.id
    assert fetched.type == PromptType.TEXT
    assert fetched.text == "find red car"
    assert fetched.frame_id is None
    assert fetched.project_id == project.id


def test_add_and_get_by_id_visual_prompt(prompt_repo, fxt_session, clean_after):
    project = make_project()
    fxt_session.add(project)
    fxt_session.commit()

    label = LabelDB(name="test", color="#FF0000", project_id=project.id)
    fxt_session.add(label)
    fxt_session.commit()

    frame_id = uuid4()
    prompt = make_visual_prompt(project.id, frame_id=frame_id)
    annotation = make_annotation(prompt.id, label_id=label.id)
    prompt.annotations.append(annotation)
    prompt_repo.add(prompt)
    fxt_session.commit()

    fetched = prompt_repo.get_by_id(prompt.id)
    assert fetched is not None
    assert fetched.type == PromptType.VISUAL
    assert fetched.frame_id == frame_id
    assert fetched.text is None
    assert len(fetched.annotations) == 1


def test_get_by_id_not_found(prompt_repo, fxt_session, clean_after):
    project = make_project()
    fxt_session.add(project)
    fxt_session.commit()

    prompt = make_text_prompt(project.id)
    prompt_repo.add(prompt)
    fxt_session.commit()

    assert prompt_repo.get_by_id(uuid4()) is None


def test_get_by_id_and_project(prompt_repo, fxt_session, clean_after):
    project_a = make_project("A")
    project_b = make_project("B")
    fxt_session.add_all([project_a, project_b])
    fxt_session.commit()

    prompt_a = make_text_prompt(project_a.id)
    prompt_b = make_text_prompt(project_b.id)
    prompt_repo.add(prompt_a)
    prompt_repo.add(prompt_b)
    fxt_session.commit()

    assert prompt_repo.get_by_id_and_project(prompt_a.id, project_a.id) is not None
    assert prompt_repo.get_by_id_and_project(prompt_a.id, project_b.id) is None


def test_get_all_by_project(prompt_repo, fxt_session, clean_after):
    project_main = make_project("main")
    project_other = make_project("other")
    fxt_session.add_all([project_main, project_other])
    fxt_session.commit()

    text_prompt = make_text_prompt(project_main.id)
    visual_prompt_1 = make_visual_prompt(project_main.id)
    visual_prompt_2 = make_visual_prompt(project_main.id)
    other_prompt = make_visual_prompt(project_other.id)

    for p in [text_prompt, visual_prompt_1, visual_prompt_2]:
        prompt_repo.add(p)
    prompt_repo.add(other_prompt)
    fxt_session.commit()

    result = prompt_repo.list_all_by_project(project_main.id)
    assert len(result) == 3
    assert {p.id for p in result} == {text_prompt.id, visual_prompt_1.id, visual_prompt_2.id}


def test_get_all_by_project_with_type_filter(prompt_repo, fxt_session, clean_after):
    project = make_project()
    fxt_session.add(project)
    fxt_session.commit()

    text_prompt = make_text_prompt(project.id)
    visual_prompt_1 = make_visual_prompt(project.id)
    visual_prompt_2 = make_visual_prompt(project.id)

    for p in [text_prompt, visual_prompt_1, visual_prompt_2]:
        prompt_repo.add(p)
    fxt_session.commit()

    all_prompts = prompt_repo.list_all_by_project(project.id)
    assert len(all_prompts) == 3

    text_prompts = prompt_repo.list_all_by_project(project.id, prompt_type=PromptType.TEXT)
    assert len(text_prompts) == 1
    assert text_prompts[0].id == text_prompt.id
    assert text_prompts[0].type == PromptType.TEXT

    visual_prompts = prompt_repo.list_all_by_project(project.id, prompt_type=PromptType.VISUAL)
    assert len(visual_prompts) == 2
    assert {p.id for p in visual_prompts} == {visual_prompt_1.id, visual_prompt_2.id}
    assert all(p.type == PromptType.VISUAL for p in visual_prompts)


def test_get_text_prompt_by_project(prompt_repo, fxt_session, clean_after):
    project = make_project()
    fxt_session.add(project)
    fxt_session.commit()

    text_prompt = make_text_prompt(project.id, text="search query")
    visual_prompt = make_visual_prompt(project.id)
    prompt_repo.add(text_prompt)
    prompt_repo.add(visual_prompt)
    fxt_session.commit()

    fetched = prompt_repo.get_text_prompt_by_project(project.id)
    assert fetched is not None
    assert fetched.id == text_prompt.id
    assert fetched.type == PromptType.TEXT
    assert fetched.text == "search query"


def test_get_text_prompt_by_project_none(prompt_repo, fxt_session, clean_after):
    project = make_project()
    fxt_session.add(project)
    fxt_session.commit()

    visual_prompt = make_visual_prompt(project.id)
    prompt_repo.add(visual_prompt)
    fxt_session.commit()

    fetched = prompt_repo.get_text_prompt_by_project(project.id)
    assert fetched is None


def test_delete_prompt(prompt_repo, fxt_session, clean_after):
    project = make_project()
    fxt_session.add(project)
    fxt_session.commit()

    prompt = make_text_prompt(project.id)
    prompt_repo.add(prompt)
    fxt_session.commit()

    assert prompt_repo.get_by_id(prompt.id) is not None
    deleted = prompt_repo.delete(prompt.id)
    fxt_session.commit()
    assert deleted is True
    assert prompt_repo.get_by_id(prompt.id) is None


def test_list_with_pagination(prompt_repo, fxt_session, clean_after):
    project = make_project()
    fxt_session.add(project)
    fxt_session.commit()

    prompts = [make_visual_prompt(project.id) for _ in range(15)]
    for p in prompts:
        prompt_repo.add(p)
    fxt_session.commit()

    page1, total = prompt_repo.list_with_pagination_by_project(project.id, offset=0, limit=10)
    assert len(page1) == 10
    assert total == 15

    page2, total = prompt_repo.list_with_pagination_by_project(project.id, offset=10, limit=10)
    assert len(page2) == 5
    assert total == 15


def test_project_deletion_cascades_prompts(prompt_repo, fxt_session, clean_after):
    project = make_project()
    fxt_session.add(project)
    fxt_session.commit()

    text_prompt = make_text_prompt(project.id)
    visual_prompt = make_visual_prompt(project.id)
    prompt_repo.add(text_prompt)
    prompt_repo.add(visual_prompt)
    fxt_session.commit()

    prompt_ids = [text_prompt.id, visual_prompt.id]

    fxt_session.delete(project)
    fxt_session.commit()

    for pid in prompt_ids:
        assert prompt_repo.get_by_id(pid) is None


def test_single_text_prompt_per_project_constraint(prompt_repo, fxt_session, clean_after):
    project = make_project()
    fxt_session.add(project)
    fxt_session.commit()

    first = make_text_prompt(project.id, text="first")
    second = make_text_prompt(project.id, text="second")

    prompt_repo.add(first)
    fxt_session.commit()

    with pytest.raises(IntegrityError):
        prompt_repo.add(second)
    fxt_session.rollback()

    fetched = prompt_repo.get_text_prompt_by_project(project.id)
    assert fetched is not None
    assert fetched.text == "first"


def test_multiple_visual_prompts_allowed(prompt_repo, fxt_session, clean_after):
    project = make_project()
    fxt_session.add(project)
    fxt_session.commit()

    for i in range(5):
        prompt = make_visual_prompt(project.id, frame_id=uuid4())
        prompt_repo.add(prompt)
    fxt_session.commit()

    result = prompt_repo.list_all_by_project(project.id)
    assert len(result) == 5
    assert all(p.type == PromptType.VISUAL for p in result)


def test_prompt_content_check_constraint_text(prompt_repo, fxt_session, clean_after):
    project = make_project()
    fxt_session.add(project)
    fxt_session.commit()

    invalid_prompt = PromptDB(
        type=PromptType.TEXT,
        text=None,
        frame_id=None,
        project_id=project.id,
    )
    with pytest.raises(IntegrityError):
        prompt_repo.add(invalid_prompt)
    fxt_session.rollback()


def test_prompt_content_check_constraint_visual(prompt_repo, fxt_session, clean_after):
    project = make_project()
    fxt_session.add(project)
    fxt_session.commit()

    invalid_prompt = PromptDB(
        type=PromptType.VISUAL,
        text=None,
        frame_id=None,
        project_id=project.id,
    )
    with pytest.raises(IntegrityError):
        prompt_repo.add(invalid_prompt)
    fxt_session.rollback()


def test_prompt_content_check_constraint_mixed(prompt_repo, fxt_session, clean_after):
    project = make_project()
    fxt_session.add(project)
    fxt_session.commit()

    invalid_prompt = PromptDB(
        type=PromptType.TEXT,
        text="content",
        frame_id=uuid4(),
        project_id=project.id,
    )
    with pytest.raises(IntegrityError):
        prompt_repo.add(invalid_prompt)
    fxt_session.rollback()


def test_annotations_cascade_delete_with_prompt(prompt_repo, fxt_session, clean_after):
    project = make_project()
    fxt_session.add(project)
    fxt_session.commit()

    label = LabelDB(name="test", color="#FF0000", project_id=project.id)
    fxt_session.add(label)
    fxt_session.commit()

    prompt = make_visual_prompt(project.id)
    ann1 = make_annotation(prompt.id, label_id=label.id)
    ann2 = make_annotation(prompt.id, label_id=label.id)
    prompt.annotations.extend([ann1, ann2])
    prompt_repo.add(prompt)
    fxt_session.commit()

    annotation_ids = [ann1.id, ann2.id]

    prompt_repo.delete(prompt.id)
    fxt_session.commit()

    for ann_id in annotation_ids:
        result = fxt_session.get(AnnotationDB, ann_id)
        assert result is None


def test_unique_frame_id_constraint(prompt_repo, fxt_session, clean_after):
    """Test that each frame_id can only be used once across all visual prompts."""
    project = make_project()
    fxt_session.add(project)
    fxt_session.commit()

    frame_id = uuid4()
    first_prompt = make_visual_prompt(project.id, frame_id=frame_id)
    second_prompt = make_visual_prompt(project.id, frame_id=frame_id)

    prompt_repo.add(first_prompt)
    fxt_session.commit()

    with pytest.raises(IntegrityError):
        prompt_repo.add(second_prompt)
    fxt_session.rollback()

    # verify only first prompt exists
    all_prompts = prompt_repo.list_all_by_project(project.id)
    assert len(all_prompts) == 1
    assert all_prompts[0].frame_id == frame_id


def test_different_frames_allowed(prompt_repo, fxt_session, clean_after):
    """Test that different frame_ids can be used for different visual prompts."""
    project = make_project()
    fxt_session.add(project)
    fxt_session.commit()

    frame_id_1 = uuid4()
    frame_id_2 = uuid4()
    frame_id_3 = uuid4()

    prompt_1 = make_visual_prompt(project.id, frame_id=frame_id_1)
    prompt_2 = make_visual_prompt(project.id, frame_id=frame_id_2)
    prompt_3 = make_visual_prompt(project.id, frame_id=frame_id_3)

    for p in [prompt_1, prompt_2, prompt_3]:
        prompt_repo.add(p)
    fxt_session.commit()

    all_prompts = prompt_repo.list_all_by_project(project.id)
    assert len(all_prompts) == 3
    assert {p.frame_id for p in all_prompts} == {frame_id_1, frame_id_2, frame_id_3}
