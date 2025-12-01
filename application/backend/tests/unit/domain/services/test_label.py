# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from pydantic_extra_types.color import Color
from sqlalchemy.exc import IntegrityError

from domain.db.models import LabelDB
from domain.errors import ResourceAlreadyExistsError, ResourceNotFoundError, ResourceType
from domain.services.label import LabelService
from domain.services.schemas.label import LabelCreateSchema, LabelsListSchema, LabelUpdateSchema

PROJECT_ID = uuid4()
LABEL_ID = uuid4()


@pytest.fixture
def mock_session():
    return MagicMock()


@pytest.fixture
def mock_label_repository():
    return MagicMock()


@pytest.fixture
def mock_project_repository():
    return MagicMock()


@pytest.fixture
def mock_label():
    return LabelDB(id=LABEL_ID, project_id=PROJECT_ID, name="Original Label", color="#ff5733")


@pytest.fixture
def label_service(mock_session, mock_label_repository, mock_project_repository):
    return LabelService(
        session=mock_session, label_repository=mock_label_repository, project_repository=mock_project_repository
    )


def test_create_label(label_service, mock_label_repository, mock_project_repository):
    label_data = LabelCreateSchema(name="Test Label", id=LABEL_ID, color=None)
    mock_project_repository.get_by_id.return_value = MagicMock()

    result = label_service.create_label(PROJECT_ID, label_data)

    assert result.name == label_data.name
    mock_label_repository.add.assert_called_once()
    label_service.session.commit.assert_called_once()


def test_create_label_duplicate_name(label_service, mock_label_repository, mock_project_repository):
    label_data = LabelCreateSchema(name="Duplicate Label", id=LABEL_ID, color=None)
    mock_project_repository.get_by_id.return_value = MagicMock()

    # Mock IntegrityError for duplicate name constraint
    mock_error = IntegrityError("statement", "params", "orig")
    mock_error.orig = Exception("UNIQUE constraint failed: uq_label_name_per_project")
    label_service.session.commit.side_effect = mock_error

    with pytest.raises(ResourceAlreadyExistsError) as exc_info:
        label_service.create_label(PROJECT_ID, label_data)

    assert exc_info.value.resource_type == ResourceType.LABEL
    assert exc_info.value.field == "name"
    assert "label with the name 'duplicate label' already exists" in str(exc_info.value).lower()
    label_service.session.rollback.assert_called_once()


def test_create_label_project_not_found(label_service, mock_project_repository):
    label_data = LabelCreateSchema(name="Test Label", id=LABEL_ID, color=None)
    mock_project_repository.get_by_id.return_value = None

    with pytest.raises(ResourceNotFoundError) as exc_info:
        label_service.create_label(PROJECT_ID, label_data)

    assert exc_info.value.resource_type == ResourceType.PROJECT


def test_get_label_by_id(label_service, mock_label_repository, mock_project_repository, mock_label):
    mock_project_repository.get_by_id.return_value = MagicMock()
    mock_label_repository.get_by_id_and_project.return_value = mock_label

    result = label_service.get_label_by_id(project_id=PROJECT_ID, label_id=LABEL_ID)

    assert result.name == "Original Label"
    assert result.id == LABEL_ID
    assert result.color == "#ff5733"
    mock_label_repository.get_by_id_and_project.assert_called_once_with(LABEL_ID, PROJECT_ID)


def test_get_label_by_id_not_found(label_service, mock_label_repository, mock_project_repository):
    mock_project_repository.get_by_id.return_value = MagicMock()
    mock_label_repository.get_by_id_and_project.return_value = None

    with pytest.raises(ResourceNotFoundError):
        label_service.get_label_by_id(PROJECT_ID, LABEL_ID)


def test_get_all_labels(label_service, mock_label_repository, mock_project_repository):
    mock_project_repository.get_by_id.return_value = MagicMock()
    label_1 = LabelDB(id=uuid4(), name="Label 1", color="#ff0000", project_id=PROJECT_ID)
    label_2 = LabelDB(id=uuid4(), name="Label 2", color="#00ff00", project_id=PROJECT_ID)
    mock_label_repository.list_with_pagination_by_project.return_value = ([label_1, label_2], 2)

    result = label_service.get_all_labels(PROJECT_ID)

    assert isinstance(result, LabelsListSchema)
    assert len(result.labels) == 2
    assert result.labels[0].name == "Label 1"
    assert result.labels[1].name == "Label 2"
    mock_label_repository.list_with_pagination_by_project.assert_called_once_with(
        project_id=PROJECT_ID, offset=0, limit=20
    )


def test_get_all_labels_empty_list(label_service, mock_label_repository, mock_project_repository):
    mock_project_repository.get_by_id.return_value = MagicMock()
    mock_label_repository.list_with_pagination_by_project.return_value = ([], 0)

    result = label_service.get_all_labels(PROJECT_ID)

    assert isinstance(result, LabelsListSchema)
    assert len(result.labels) == 0
    mock_label_repository.list_with_pagination_by_project.assert_called_once_with(
        project_id=PROJECT_ID, offset=0, limit=20
    )


def test_delete_label_not_found(label_service, mock_label_repository, mock_project_repository):
    mock_project_repository.get_by_id.return_value = MagicMock()
    mock_label_repository.get_by_id_and_project.return_value = None

    with pytest.raises(ResourceNotFoundError):
        label_service.delete_label(PROJECT_ID, LABEL_ID)


def test_update_label_name_successfully(label_service, mock_label_repository, mock_project_repository, mock_label):
    mock_project_repository.get_by_id.return_value = MagicMock()
    mock_label_repository.get_by_id_and_project.return_value = mock_label
    mock_label_repository.update.return_value = mock_label
    update_data = LabelUpdateSchema(name="Updated Label", color=None)

    result = label_service.update_label(PROJECT_ID, LABEL_ID, update_data)

    assert result.name == "Updated Label"
    assert result.color == "#ff5733"
    label_service.session.commit.assert_called_once()


def test_update_label_color_successfully(label_service, mock_label_repository, mock_project_repository, mock_label):
    mock_project_repository.get_by_id.return_value = MagicMock()
    mock_label_repository.get_by_id_and_project.return_value = mock_label
    mock_label_repository.update.return_value = mock_label
    new_color = Color("#abcdef")
    update_data = LabelUpdateSchema(name=mock_label.name, color=new_color)

    result = label_service.update_label(PROJECT_ID, LABEL_ID, update_data)

    assert result.color == "#abcdef"
    label_service.session.commit.assert_called_once()


def test_update_label_both_name_and_color(label_service, mock_label_repository, mock_project_repository, mock_label):
    mock_project_repository.get_by_id.return_value = MagicMock()
    mock_label_repository.get_by_id_and_project.return_value = mock_label
    mock_label_repository.update.return_value = mock_label
    new_color = Color("#123456")
    update_data = LabelUpdateSchema(name="New Name", color=new_color)

    result = label_service.update_label(PROJECT_ID, LABEL_ID, update_data)

    assert result.name == "New Name"
    assert result.color == "#123456"
    label_service.session.commit.assert_called_once()


def test_update_label_no_changes(label_service, mock_label_repository, mock_project_repository, mock_label):
    mock_project_repository.get_by_id.return_value = MagicMock()
    mock_label_repository.get_by_id_and_project.return_value = mock_label
    mock_label_repository.update.return_value = mock_label
    update_data = LabelUpdateSchema(name="Original Label", color=Color("#ff5733"))

    result = label_service.update_label(PROJECT_ID, LABEL_ID, update_data)

    mock_label_repository.update.assert_called_once()
    # The result should still be the label schema
    assert result.name == "Original Label"
    assert result.color == "#ff5733"
