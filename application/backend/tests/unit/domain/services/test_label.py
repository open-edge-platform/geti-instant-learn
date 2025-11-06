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
from domain.services.schemas.label import LabelCreateSchema, LabelSchema, LabelsListSchema, LabelUpdateSchema

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
    label = MagicMock(spec=LabelDB)
    label.id = LABEL_ID
    label.project_id = PROJECT_ID
    label.name = "Original Label"
    label.color = "#ff5733"
    return label


@pytest.fixture
def label_service(mock_session, mock_label_repository, mock_project_repository):
    return LabelService(
        session=mock_session,
        label_repository=mock_label_repository,
        project_repository=mock_project_repository,
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


def test_get_label_by_id(label_service, mock_label_repository, mock_project_repository):
    mock_project_repository.get_by_id.return_value = MagicMock()
    mock_label = MagicMock()
    mock_label.id = LABEL_ID
    mock_label.name = "Test Label"
    mock_label.color = "#ffffff"
    mock_label_repository.get_by_id.return_value = mock_label

    result = label_service.get_label_by_id(project_id=PROJECT_ID, label_id=LABEL_ID)

    assert result.name == "Test Label"
    assert result.id == LABEL_ID
    assert result.color.upper() == "#FFFFFF"
    mock_label_repository.get_by_id.assert_called_once_with(project_id=PROJECT_ID, label_id=LABEL_ID)


def test_get_label_by_id_not_found(label_service, mock_label_repository, mock_project_repository):
    mock_project_repository.get_by_id.return_value = MagicMock()
    mock_label_repository.get_by_id.return_value = None

    with pytest.raises(ResourceNotFoundError):
        label_service.get_label_by_id(PROJECT_ID, LABEL_ID)


def test_get_all_labels(label_service, mock_label_repository, mock_project_repository):
    mock_project_repository.get_by_id.return_value = MagicMock()
    label_1 = MagicMock(spec=LabelSchema)
    label_1.id = uuid4()
    label_1.name = "Label 1"
    label_1.color = "#ff0000"
    label_2 = MagicMock(spec=LabelSchema)
    label_2.id = uuid4()
    label_2.name = "Label 2"
    label_2.color = "#00ff00"
    mock_label_repository.get_paginated.return_value = ([label_1, label_2], 2)

    result = label_service.get_all_labels(PROJECT_ID)

    assert isinstance(result, LabelsListSchema)
    assert len(result.labels) == 2
    mock_label_repository.get_paginated.assert_called_once()


def test_get_all_labels_empty_list(label_service, mock_label_repository, mock_project_repository):
    mock_project_repository.get_by_id.return_value = MagicMock()
    mock_label_repository.get_paginated.return_value = ([], 0)

    result = label_service.get_all_labels(PROJECT_ID)

    assert isinstance(result, LabelsListSchema)
    assert len(result.labels) == 0
    mock_label_repository.get_paginated.assert_called_once()


def test_delete_label(label_service, mock_label_repository, mock_project_repository):
    mock_project_repository.get_by_id.return_value = MagicMock()
    mock_label_repository.get_by_id.return_value = MagicMock()

    label_service.delete_label(PROJECT_ID, LABEL_ID)

    mock_label_repository.delete.assert_called_once()
    label_service.session.commit.assert_called_once()


def test_delete_label_not_found(label_service, mock_label_repository, mock_project_repository):
    mock_project_repository.get_by_id.return_value = MagicMock()
    mock_label_repository.get_by_id.return_value = None

    with pytest.raises(ResourceNotFoundError):
        label_service.delete_label(PROJECT_ID, LABEL_ID)


def test_update_label_name_successfully(label_service, mock_label_repository, mock_project_repository, mock_label):
    mock_project_repository.get_by_id.return_value = MagicMock()
    mock_label_repository.get_by_id.return_value = mock_label
    update_data = LabelUpdateSchema(name="Updated Label", color=None)

    result = label_service.update_label(PROJECT_ID, LABEL_ID, update_data)

    assert result.name == "Updated Label"
    label_service.session.commit.assert_called_once()
    label_service.session.refresh.assert_called_once_with(mock_label)


def test_update_label_color_successfully(label_service, mock_label_repository, mock_project_repository, mock_label):
    mock_project_repository.get_by_id.return_value = MagicMock()
    mock_label_repository.get_by_id.return_value = mock_label
    new_color = Color("#abcdef")
    update_data = LabelUpdateSchema(name=mock_label.name, color=new_color)

    result = label_service.update_label(PROJECT_ID, LABEL_ID, update_data)

    assert result.color.upper() == "#ABCDEF"
    label_service.session.commit.assert_called_once()
    label_service.session.refresh.assert_called_once_with(mock_label)


def test_update_label_both_name_and_color(label_service, mock_label_repository, mock_project_repository, mock_label):
    mock_project_repository.get_by_id.return_value = MagicMock()
    mock_label_repository.get_by_id.return_value = mock_label
    new_color = Color("#123456")
    update_data = LabelUpdateSchema(name="New Name", color=new_color)

    result = label_service.update_label(PROJECT_ID, LABEL_ID, update_data)

    assert result.name == "New Name"
    assert result.color.upper() == "#123456"
    label_service.session.commit.assert_called_once()
    label_service.session.refresh.assert_called_once_with(mock_label)


def test_update_label_no_changes(label_service, mock_label_repository, mock_project_repository, mock_label):
    mock_project_repository.get_by_id.return_value = MagicMock()
    mock_label_repository.get_by_id.return_value = mock_label
    mock_label.name = "Original Label"
    mock_label.color = "#ff5733"
    update_data = LabelUpdateSchema(name="Original Label", color=Color("#FF5733"))

    label_service.update_label(PROJECT_ID, LABEL_ID, update_data)

    label_service.session.commit.assert_not_called()
    label_service.session.refresh.assert_not_called()


def test_update_label_project_not_found(label_service, mock_label_repository, mock_project_repository):
    mock_project_repository.get_by_id.return_value = None
    update_data = LabelUpdateSchema(name="New Name", color=None)
    other_project_id = uuid4()

    with pytest.raises(ResourceNotFoundError) as exc_info:
        label_service.update_label(other_project_id, LABEL_ID, update_data)
    assert exc_info.value.resource_type == ResourceType.PROJECT


def test_update_label_label_not_found(label_service, mock_label_repository, mock_project_repository):
    mock_project_repository.get_by_id.return_value = MagicMock()
    mock_label_repository.get_by_id.return_value = None
    update_data = LabelUpdateSchema(name="New Name", color=None)
    other_label_id = uuid4()

    with pytest.raises(ResourceNotFoundError) as exc_info:
        label_service.update_label(PROJECT_ID, other_label_id, update_data)
    assert exc_info.value.resource_type == ResourceType.LABEL


def test_update_label_duplicate_name(label_service, mock_label_repository, mock_project_repository, mock_label):
    mock_project_repository.get_by_id.return_value = MagicMock()
    mock_label_repository.get_by_id.return_value = mock_label
    update_data = LabelUpdateSchema(name="Duplicate Name", color=None)

    # Mock IntegrityError for duplicate name constraint
    mock_error = IntegrityError("statement", "params", "orig")
    mock_error.orig = Exception("UNIQUE constraint failed: uq_label_name_per_project")
    label_service.session.commit.side_effect = mock_error

    with pytest.raises(ResourceAlreadyExistsError) as exc_info:
        label_service.update_label(PROJECT_ID, LABEL_ID, update_data)

    assert exc_info.value.resource_type == ResourceType.LABEL
    assert exc_info.value.field == "name"
    assert "label with the name 'duplicate name' already exists" in str(exc_info.value).lower()
    label_service.session.rollback.assert_called_once()
