# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError

from services.errors import ResourceAlreadyExistsError, ResourceNotFoundError
from services.label import LabelService
from services.schemas.label import LabelCreateSchema, LabelSchema, LabelsListSchema


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
def mock_dispatcher():
    return MagicMock()


@pytest.fixture
def label_service(mock_session, mock_label_repository, mock_project_repository, mock_dispatcher):
    return LabelService(
        session=mock_session,
        label_repository=mock_label_repository,
        project_repository=mock_project_repository,
        config_change_dispatcher=mock_dispatcher,
    )


def test_create_label(label_service, mock_label_repository, mock_project_repository):
    project_id = uuid4()
    label_id = uuid4()
    label_data = LabelCreateSchema(name="Test Label", id=label_id, color=None)
    mock_project_repository.get_by_id.return_value = MagicMock()
    mock_label_repository.exists_by_name.return_value = False
    mock_label_repository.exists_by_id.return_value = False

    result = label_service.create_label(project_id, label_data)

    assert result.name == label_data.name
    mock_label_repository.add.assert_called_once()
    label_service.session.commit.assert_called_once()


def test_create_label_duplicate_name(label_service, mock_label_repository, mock_project_repository):
    project_id = uuid4()
    label_id = uuid4()
    label_data = LabelCreateSchema(name="Duplicate Label", id=label_id, color=None)
    mock_project_repository.get_by_id.return_value = MagicMock()

    # Mock IntegrityError for duplicate name constraint
    mock_error = IntegrityError("statement", "params", "orig")
    mock_error.orig = Exception("UNIQUE constraint failed: label_name_project_unique")
    label_service.session.commit.side_effect = mock_error

    with pytest.raises(ResourceAlreadyExistsError):
        label_service.create_label(project_id, label_data)

    label_service.session.rollback.assert_called_once()


def test_create_label_duplicate_id(label_service, mock_label_repository, mock_project_repository):
    project_id = uuid4()
    label_id = uuid4()
    label_data = LabelCreateSchema(name="Duplicate Label", id=label_id, color=None)
    mock_project_repository.get_by_id.return_value = MagicMock()

    # Mock IntegrityError for duplicate primary key
    mock_error = IntegrityError("statement", "params", "orig")
    mock_error.orig = Exception("UNIQUE constraint failed: primary key")
    label_service.session.commit.side_effect = mock_error

    with pytest.raises(ResourceAlreadyExistsError):
        label_service.create_label(project_id, label_data)

    label_service.session.rollback.assert_called_once()


def test_get_label_by_id(label_service, mock_label_repository, mock_project_repository):
    project_id = uuid4()
    label_id = uuid4()
    mock_project_repository.get_by_id.return_value = MagicMock()
    mock_label = MagicMock()
    mock_label.id = label_id
    mock_label.name = "Test Label"
    mock_label.color = "#FFFFFF"
    mock_label_repository.get_by_id.return_value = mock_label

    result = label_service.get_label_by_id(project_id=project_id, label_id=label_id)

    assert result.name == "Test Label"
    assert result.id == label_id
    assert result.color.upper() == "#FFFFFF"
    mock_label_repository.get_by_id.assert_called_once_with(project_id=project_id, label_id=label_id)


def test_get_label_by_id_not_found(label_service, mock_label_repository, mock_project_repository):
    project_id = uuid4()
    label_id = uuid4()
    mock_project_repository.get_by_id.return_value = MagicMock()
    mock_label_repository.get_by_id.return_value = None

    with pytest.raises(ResourceNotFoundError):
        label_service.get_label_by_id(project_id, label_id)


def test_get_all_labels(label_service, mock_label_repository, mock_project_repository):
    project_id = uuid4()
    mock_project_repository.get_by_id.return_value = MagicMock()
    label_1 = MagicMock(spec=LabelSchema)
    label_1.id = uuid4()
    label_1.name = "Label 1"
    label_1.color = "#FF0000"
    label_2 = MagicMock(spec=LabelSchema)
    label_2.id = uuid4()
    label_2.name = "Label 2"
    label_2.color = "#00FF00"
    mock_label_repository.get_paginated.return_value = ([label_1, label_2], 0)

    result = label_service.get_all_labels(project_id)

    assert isinstance(result, LabelsListSchema)
    assert len(result.labels) == 2
    mock_label_repository.get_paginated.assert_called_once()


def test_get_all_labels_empty_list(label_service, mock_label_repository, mock_project_repository):
    project_id = uuid4()
    mock_project_repository.get_by_id.return_value = MagicMock()
    mock_label_repository.get_paginated.return_value = ([], 0)

    result = label_service.get_all_labels(project_id)

    assert isinstance(result, LabelsListSchema)
    assert len(result.labels) == 0
    mock_label_repository.get_paginated.assert_called_once()


def test_delete_label(label_service, mock_label_repository, mock_project_repository):
    project_id = uuid4()
    label_id = uuid4()
    mock_project_repository.get_by_id.return_value = MagicMock()
    mock_label_repository.get_by_id.return_value = MagicMock()

    label_service.delete_label(project_id, label_id)

    mock_label_repository.delete.assert_called_once()
    label_service.session.commit.assert_called_once()


def test_delete_label_not_found(label_service, mock_label_repository, mock_project_repository):
    project_id = uuid4()
    label_id = uuid4()
    mock_project_repository.get_by_id.return_value = MagicMock()
    mock_label_repository.get_by_id.return_value = None

    with pytest.raises(ResourceNotFoundError):
        label_service.delete_label(project_id, label_id)
