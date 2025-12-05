# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError

from domain.db.models import AnnotationDB, LabelDB, ProjectDB, PromptDB, PromptType
from domain.repositories.annotation import AnnotationRepository


@pytest.fixture
def annotation_repo(fxt_session):
    return AnnotationRepository(session=fxt_session)


@pytest.fixture
def clean_after(fxt_clean_table):
    yield
    fxt_clean_table(AnnotationDB)
    fxt_clean_table(PromptDB)
    fxt_clean_table(LabelDB)
    fxt_clean_table(ProjectDB)


def make_project(name=None) -> ProjectDB:
    return ProjectDB(name=name or f"proj-{uuid4().hex[:8]}")


def make_label(project_id, name="test-label", color="#FF0000") -> LabelDB:
    return LabelDB(name=name, color=color, project_id=project_id)


def make_prompt(project_id, frame_id=None) -> PromptDB:
    frame_id = frame_id or uuid4()
    return PromptDB(
        type=PromptType.VISUAL,
        text=None,
        frame_id=frame_id,
        project_id=project_id,
    )


def make_annotation(prompt_id, label_id, config=None) -> AnnotationDB:
    if config is None:
        config = {"type": "rectangle", "points": [{"x": 0.1, "y": 0.1}, {"x": 0.5, "y": 0.5}]}
    return AnnotationDB(
        config=config,
        label_id=label_id,
        prompt_id=prompt_id,
    )


def test_add_annotation_with_label(annotation_repo, fxt_session, clean_after):
    """Test adding an annotation with a label reference."""
    project = make_project()
    fxt_session.add(project)
    fxt_session.commit()

    label = make_label(project.id, name="car")
    fxt_session.add(label)
    fxt_session.commit()

    prompt = make_prompt(project.id)
    fxt_session.add(prompt)
    fxt_session.commit()

    annotation = make_annotation(prompt.id, label_id=label.id)
    annotation_repo.add(annotation)
    fxt_session.commit()

    fetched = annotation_repo.get_by_id(annotation.id)
    assert fetched is not None
    assert fetched.label_id == label.id
    assert fetched.prompt_id == prompt.id
    assert fetched.config["type"] == "rectangle"


def test_label_deletion_restricted_when_in_use(annotation_repo, fxt_session, clean_after):
    """Test that deleting a label fails when it's referenced by annotations (ondelete=RESTRICT)."""
    project = make_project()
    fxt_session.add(project)
    fxt_session.commit()

    label = make_label(project.id)
    fxt_session.add(label)
    fxt_session.commit()

    prompt = make_prompt(project.id)
    fxt_session.add(prompt)
    fxt_session.commit()

    annotation = make_annotation(prompt.id, label_id=label.id)
    annotation_repo.add(annotation)
    fxt_session.commit()

    annotation_id = annotation.id
    assert annotation.label_id == label.id

    # Try to delete the label - should fail with IntegrityError
    with pytest.raises(IntegrityError):
        fxt_session.delete(label)
        fxt_session.commit()
    fxt_session.rollback()

    # Verify annotation still exists and label_id is unchanged
    fetched = annotation_repo.get_by_id(annotation_id)
    assert fetched is not None
    assert fetched.label_id == label.id


def test_label_deletion_succeeds_after_annotation_deleted(annotation_repo, fxt_session, clean_after):
    """Test that label can be deleted after all referencing annotations are deleted."""
    project = make_project()
    fxt_session.add(project)
    fxt_session.commit()

    label = make_label(project.id)
    fxt_session.add(label)
    fxt_session.commit()

    prompt = make_prompt(project.id)
    fxt_session.add(prompt)
    fxt_session.commit()

    annotation = make_annotation(prompt.id, label_id=label.id)
    annotation_repo.add(annotation)
    fxt_session.commit()

    label_id = label.id

    # Delete annotation first
    annotation_repo.delete(annotation.id)
    fxt_session.commit()

    # Now label deletion should succeed
    fxt_session.delete(label)
    fxt_session.commit()

    # Verify label is deleted
    assert fxt_session.get(LabelDB, label_id) is None


def test_is_label_in_use_true(annotation_repo, fxt_session, clean_after):
    """Test is_label_in_use returns True when label is referenced."""
    project = make_project()
    fxt_session.add(project)
    fxt_session.commit()

    label = make_label(project.id)
    fxt_session.add(label)
    fxt_session.commit()

    prompt = make_prompt(project.id)
    fxt_session.add(prompt)
    fxt_session.commit()

    annotation = make_annotation(prompt.id, label_id=label.id)
    annotation_repo.add(annotation)
    fxt_session.commit()

    assert annotation_repo.is_label_in_use(label.id) is True


def test_is_label_in_use_false(annotation_repo, fxt_session, clean_after):
    """Test is_label_in_use returns False when label is not referenced."""
    project = make_project()
    fxt_session.add(project)
    fxt_session.commit()

    label = make_label(project.id)
    fxt_session.add(label)
    fxt_session.commit()

    assert annotation_repo.is_label_in_use(label.id) is False


def test_is_label_in_use_after_annotation_deletion(annotation_repo, fxt_session, clean_after):
    """Test is_label_in_use returns False after annotation is deleted."""
    project = make_project()
    fxt_session.add(project)
    fxt_session.commit()

    label = make_label(project.id)
    fxt_session.add(label)
    fxt_session.commit()

    prompt = make_prompt(project.id)
    fxt_session.add(prompt)
    fxt_session.commit()

    annotation = make_annotation(prompt.id, label_id=label.id)
    annotation_repo.add(annotation)
    fxt_session.commit()

    assert annotation_repo.is_label_in_use(label.id) is True

    annotation_repo.delete(annotation.id)
    fxt_session.commit()

    assert annotation_repo.is_label_in_use(label.id) is False


def test_is_label_in_use_with_multiple_annotations(annotation_repo, fxt_session, clean_after):
    """Test is_label_in_use with multiple annotations using the same label."""
    project = make_project()
    fxt_session.add(project)
    fxt_session.commit()

    label = make_label(project.id)
    fxt_session.add(label)
    fxt_session.commit()

    prompt1 = make_prompt(project.id)
    prompt2 = make_prompt(project.id)
    fxt_session.add_all([prompt1, prompt2])
    fxt_session.commit()

    ann1 = make_annotation(prompt1.id, label_id=label.id)
    ann2 = make_annotation(prompt2.id, label_id=label.id)
    annotation_repo.add(ann1)
    annotation_repo.add(ann2)
    fxt_session.commit()

    assert annotation_repo.is_label_in_use(label.id) is True

    # Delete one annotation - label still in use
    annotation_repo.delete(ann1.id)
    fxt_session.commit()
    assert annotation_repo.is_label_in_use(label.id) is True

    # Delete second annotation - label no longer in use
    annotation_repo.delete(ann2.id)
    fxt_session.commit()
    assert annotation_repo.is_label_in_use(label.id) is False


def test_annotation_cascade_delete_with_prompt(annotation_repo, fxt_session, clean_after):
    """Test that annotations are deleted when their prompt is deleted (ondelete=CASCADE)."""
    project = make_project()
    fxt_session.add(project)
    fxt_session.commit()

    label = make_label(project.id)
    fxt_session.add(label)
    fxt_session.commit()

    prompt = make_prompt(project.id)
    fxt_session.add(prompt)
    fxt_session.commit()

    ann1 = make_annotation(prompt.id, label_id=label.id)
    ann2 = make_annotation(prompt.id, label_id=label.id)
    annotation_repo.add(ann1)
    annotation_repo.add(ann2)
    fxt_session.commit()

    annotation_ids = [ann1.id, ann2.id]

    # Delete the prompt
    fxt_session.delete(prompt)
    fxt_session.commit()

    # Verify annotations are deleted
    for ann_id in annotation_ids:
        assert annotation_repo.get_by_id(ann_id) is None


def test_annotation_foreign_key_constraints(annotation_repo, fxt_session, clean_after):
    """Test that annotation requires valid prompt_id and label_id."""
    project = make_project()
    fxt_session.add(project)
    fxt_session.commit()

    label = make_label(project.id)
    fxt_session.add(label)
    fxt_session.commit()

    # Try to create annotation with non-existent prompt_id
    invalid_annotation = make_annotation(prompt_id=uuid4(), label_id=label.id)

    with pytest.raises(IntegrityError):
        annotation_repo.add(invalid_annotation)
        fxt_session.commit()
    fxt_session.rollback()

    # Try to create annotation with non-existent label_id
    prompt = make_prompt(project.id)
    fxt_session.add(prompt)
    fxt_session.commit()

    invalid_annotation2 = make_annotation(prompt_id=prompt.id, label_id=uuid4())

    with pytest.raises(IntegrityError):
        annotation_repo.add(invalid_annotation2)
        fxt_session.commit()
    fxt_session.rollback()


def test_update_annotation_label(annotation_repo, fxt_session, clean_after):
    """Test updating annotation's label reference."""
    project = make_project()
    fxt_session.add(project)
    fxt_session.commit()

    label1 = make_label(project.id, name="car")
    label2 = make_label(project.id, name="truck")
    fxt_session.add_all([label1, label2])
    fxt_session.commit()

    prompt = make_prompt(project.id)
    fxt_session.add(prompt)
    fxt_session.commit()

    annotation = make_annotation(prompt.id, label_id=label1.id)
    annotation_repo.add(annotation)
    fxt_session.commit()

    # Update to different label
    annotation.label_id = label2.id
    annotation_repo.update(annotation)
    fxt_session.commit()

    fetched = annotation_repo.get_by_id(annotation.id)
    assert fetched.label_id == label2.id
