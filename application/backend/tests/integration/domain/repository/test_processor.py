# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from domain.db.models import ProcessorDB
from domain.repositories.processor import ProcessorRepository


@pytest.fixture
def mock_session():
    """Create a mock SQLAlchemy session."""
    return Mock(spec=Session)


@pytest.fixture
def repository(mock_session):
    """Create a ProcessorRepository instance with a mock session."""
    return ProcessorRepository(mock_session)


@pytest.fixture
def sample_processor():
    """Create a sample ProcessorDB entity."""
    return ProcessorDB(id=uuid4(), project_id=uuid4(), active=False)


class TestProcessorRepository:
    """Test suite for ProcessorRepository."""

    def test_init(self, mock_session):
        """Test repository initialization."""
        repo = ProcessorRepository(mock_session)
        assert repo.session == mock_session

    def test_add(self, repository, mock_session, sample_processor):
        """Test adding a processor to the session."""
        repository.add(sample_processor)
        mock_session.add.assert_called_once_with(sample_processor)

    def test_get_by_id(self, repository, mock_session, sample_processor):
        """Test retrieving a processor by ID."""
        mock_session.scalars.return_value.first.return_value = sample_processor

        result = repository.get_by_id(sample_processor.id)

        assert result == sample_processor
        mock_session.scalars.assert_called_once()

    def test_get_by_id_not_found(self, repository, mock_session):
        """Test retrieving a non-existent processor."""
        mock_session.scalars.return_value.first.return_value = None

        result = repository.get_by_id(uuid4())

        assert result is None

    def test_get_by_id_and_project(self, repository, mock_session, sample_processor):
        """Test retrieving a processor by ID and project."""
        mock_session.scalars.return_value.first.return_value = sample_processor

        result = repository.get_by_id_and_project(sample_processor.id, sample_processor.project_id)

        assert result == sample_processor
        mock_session.scalars.assert_called_once()

    def test_get_all_by_project(self, repository, mock_session, sample_processor):
        """Test retrieving all processors for a project."""
        processors = [sample_processor, ProcessorDB(id=uuid4(), project_id=sample_processor.project_id)]
        mock_session.scalars.return_value.all.return_value = processors

        result = repository.get_all_by_project(sample_processor.project_id)

        assert result == processors
        assert len(result) == 2
        mock_session.scalars.assert_called_once()

    def test_get_all_by_project_empty(self, repository, mock_session):
        """Test retrieving processors for a project with no processors."""
        mock_session.scalars.return_value.all.return_value = []

        result = repository.get_all_by_project(uuid4())

        assert result == []

    def test_delete(self, repository, mock_session, sample_processor):
        """Test deleting a processor from the session."""
        repository.delete(sample_processor)
        mock_session.delete.assert_called_once_with(sample_processor)

    def test_get_activated_in_project(self, repository, mock_session):
        """Test retrieving the active processor in a project."""
        project_id = uuid4()
        active_processor = ProcessorDB(id=uuid4(), project_id=project_id, active=True)
        mock_session.scalars.return_value.first.return_value = active_processor

        result = repository.get_activated_in_project(project_id)

        assert result == active_processor
        mock_session.scalars.assert_called_once()

    def test_get_activated_in_project_none(self, repository, mock_session):
        """Test retrieving active processor when none exists."""
        mock_session.scalars.return_value.first.return_value = None

        result = repository.get_activated_in_project(uuid4())

        assert result is None
